#!/usr/bin/env bash
# Stand up the EFS proof: a real ECS fleet on real EFS, for the simulator to attach to.
#
# The same shape as `deploy/attached-local/up.sh` — pre-existing storage, replicas somebody else
# runs, faults through a script — with ECS where that one has processes and EFS where it has an NFS
# export. That is the point: the *same* nine scenarios grade both, so a claim proved on the homelab
# and a claim proved here are the same claim.
#
# Idempotent throughout: every resource is looked up by its `fleetsim=efs-proof` tag before being
# created, so running this twice adds nothing and a half-finished run can simply be rerun.
#
#     ./up.sh            # provision, build nothing, launch the fleet
#     ./up.sh --images   # also build and push the two images first
#     ./down.sh          # give it all back, and verify that it did
#
# ## Two AWS facts this encodes
#
# **One AZ.** EFS mount targets are per-AZ and cross-AZ traffic costs $0.01/GB each way *and* adds
# latency to the replay path — which is precisely what the takeover measurement is trying to isolate.
# So every subnet is in one AZ and there is one mount target.
#
# **Elastic throughput, not the Bursting default.** The soak writes ~933 bytes per turn, so at the
# homelab's 190 turns/s it sustains ~178 KB/s. Bursting's baseline scales with *stored* size and this
# filesystem holds ~0.2 GB: it would spend burst credits and then throttle, making the throughput
# number a reading of EFS's credit balance rather than of the design.
#
# ## Why the replica task definition overrides the image's entrypoint
#
# `Dockerfile.agent` execs the agent, so the agent is PID 1 — correct for production, and fatal for
# the `kill` fault, because the kernel refuses SIGKILL to a PID namespace's init from inside it. The
# override below backgrounds the agent under a shell that forwards SIGTERM, so `term` is still a real
# graceful stop and `kill` is a real hard one. The shipped image is untouched.
set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
AZ=${FLEETSIM_AZ:-${REGION}a}
CLUSTER=${FLEETSIM_CLUSTER:-fleetsim}
REPLICAS=${REPLICAS:-3}
CIDR=${FLEETSIM_CIDR:-10.77.0.0/16}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
TAG="Name=tag:fleetsim,Values=efs-proof"
TAGSPEC='{Key=fleetsim,Value=efs-proof}'
aws() { command aws --region "$REGION" --output text "$@"; }
ACCOUNT=$(aws sts get-caller-identity --query Account)
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

say() { echo "── $* ──"; }

# The deterministic grant material. `Minter::new`'s seeds are fixed so a failure reproduces, which
# also means these are **not secret** — an attached fleet must be somewhere private. Taken from the
# simulator rather than written down here, so the two can never disagree.
eval "$("$ROOT/target/debug/fleet-sim" keys --dir "$(mktemp -d)")"
GRANT_KEY=$AI_AGENT_GRANT_KEY
SEAL_B64=$(cat "$AI_AGENT_SEAL_KEY")

if [ "${1:-}" = "--images" ]; then
  say "images"
  aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR" >/dev/null
  for r in fleetsim-agent fleetsim-driver; do
    aws ecr describe-repositories --repository-names "$r" >/dev/null 2>&1 \
      || aws ecr create-repository --repository-name "$r" --tags "Key=fleetsim,Value=efs-proof" >/dev/null
  done
  docker build -f "$ROOT/Dockerfile.agent"  -t "$ECR/fleetsim-agent:latest"  "$ROOT"
  docker push "$ECR/fleetsim-agent:latest"
  docker build -f "$ROOT/Dockerfile.driver" -t "$ECR/fleetsim-driver:latest" "$ROOT"
  docker push "$ECR/fleetsim-driver:latest"
fi

say "vpc"
VPC=$(aws ec2 describe-vpcs --filters "$TAG" --query 'Vpcs[0].VpcId')
if [ "$VPC" = "None" ] || [ -z "$VPC" ]; then
  VPC=$(aws ec2 create-vpc --cidr-block "$CIDR" \
    --tag-specifications "ResourceType=vpc,Tags=[$TAGSPEC]" --query 'Vpc.VpcId')
fi
# Both, and both needed: the EFS mount helper resolves `fs-….efs.<region>.amazonaws.com`, and
# without DNS hostnames every task dies in `ResourceInitializationError: Failed to resolve`.
aws ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-support
aws ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames
echo "vpc $VPC"

say "subnets"
# One /28 per replica plus one for the driver. Per-replica subnets are what makes a partition
# possible at all: a NACL is evaluated at a subnet boundary, so a replica sharing a subnet with the
# mount target could not be cut off from it. The mount target lives in the driver's subnet, which is
# also why the driver's own mount is never affected by a replica's partition.
SUBNETS=()
for i in $(seq 0 "$REPLICAS"); do
  want="10.77.$i.0/28"
  id=$(aws ec2 describe-subnets --filters "$TAG" "Name=cidr-block,Values=$want" --query 'Subnets[0].SubnetId')
  if [ "$id" = "None" ] || [ -z "$id" ]; then
    id=$(aws ec2 create-subnet --vpc-id "$VPC" --cidr-block "$want" --availability-zone "$AZ" \
      --tag-specifications "ResourceType=subnet,Tags=[$TAGSPEC]" --query 'Subnet.SubnetId')
  fi
  SUBNETS+=("$id")
done
DRIVER_SUBNET=${SUBNETS[$REPLICAS]}
echo "subnets ${SUBNETS[*]} (driver: $DRIVER_SUBNET)"

say "routing"
# Public IPs over an internet gateway rather than a NAT: the tasks need ECR, CloudWatch and the ECS
# and EC2 APIs, and for a fleet that exists for an afternoon this is one gateway instead of a NAT's
# hourly charge plus an elastic IP. Inbound is closed by the security group regardless.
IGW=$(aws ec2 describe-internet-gateways --filters "$TAG" --query 'InternetGateways[0].InternetGatewayId')
if [ "$IGW" = "None" ] || [ -z "$IGW" ]; then
  IGW=$(aws ec2 create-internet-gateway --tag-specifications "ResourceType=internet-gateway,Tags=[$TAGSPEC]" \
    --query 'InternetGateway.InternetGatewayId')
  aws ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"
fi
RT=$(aws ec2 describe-route-tables --filters "$TAG" --query 'RouteTables[0].RouteTableId')
if [ "$RT" = "None" ] || [ -z "$RT" ]; then
  RT=$(aws ec2 create-route-table --vpc-id "$VPC" \
    --tag-specifications "ResourceType=route-table,Tags=[$TAGSPEC]" --query 'RouteTable.RouteTableId')
  aws ec2 create-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null
fi
for s in "${SUBNETS[@]}"; do
  aws ec2 associate-route-table --route-table-id "$RT" --subnet-id "$s" >/dev/null 2>&1 || true
  aws ec2 modify-subnet-attribute --subnet-id "$s" --map-public-ip-on-launch 2>/dev/null || true
done

say "security groups"
sg_for() {
  local name=$1 desc=$2 id
  id=$(aws ec2 describe-security-groups --filters "$TAG" "Name=group-name,Values=$name" --query 'SecurityGroups[0].GroupId')
  if [ "$id" = "None" ] || [ -z "$id" ]; then
    id=$(aws ec2 create-security-group --group-name "$name" --description "$desc" --vpc-id "$VPC" \
      --tag-specifications "ResourceType=security-group,Tags=[$TAGSPEC]" --query 'GroupId')
  fi
  echo "$id"
}
SG_TASKS=$(sg_for fleetsim-tasks "fleetsim tasks")
SG_EFS=$(sg_for fleetsim-efs "fleetsim efs mount target")
aws ec2 authorize-security-group-ingress --group-id "$SG_TASKS" --protocol tcp --port 0-65535 --cidr "$CIDR" >/dev/null 2>&1 || true
aws ec2 authorize-security-group-ingress --group-id "$SG_EFS" --protocol tcp --port 2049 --source-group "$SG_TASKS" >/dev/null 2>&1 || true

say "network acls"
# One per replica subnet, allow-all by default. A partition is then a single deny rule at a lower
# rule number, and a heal is deleting it. NACLs and not security groups because NACLs are
# **stateless**: they are evaluated per packet, so an already-established NFS connection is cut
# immediately. A security group is stateful, and whether revoking a rule tears down a tracked flow is
# exactly the kind of thing that yields "the design strands sessions" results that are really rig
# artifacts.
for i in $(seq 1 "$REPLICAS"); do
  sub=${SUBNETS[$((i - 1))]}
  acl=$(aws ec2 describe-network-acls --filters "$TAG" "Name=tag:replica,Values=r$i" --query 'NetworkAcls[0].NetworkAclId')
  if [ "$acl" = "None" ] || [ -z "$acl" ]; then
    acl=$(aws ec2 create-network-acl --vpc-id "$VPC" \
      --tag-specifications "ResourceType=network-acl,Tags=[$TAGSPEC,{Key=replica,Value=r$i}]" \
      --query 'NetworkAcl.NetworkAclId')
    for dir in --egress --ingress; do
      aws ec2 create-network-acl-entry --network-acl-id "$acl" --rule-number 100 --protocol -1 \
        --rule-action allow $dir --cidr-block 0.0.0.0/0
    done
    assoc=$(aws ec2 describe-network-acls --filters "Name=association.subnet-id,Values=$sub" \
      --query "NetworkAcls[0].Associations[?SubnetId=='$sub'].NetworkAclAssociationId")
    aws ec2 replace-network-acl-association --association-id "$assoc" --network-acl-id "$acl" >/dev/null
  fi
done

say "efs"
FS=$(aws efs describe-file-systems --query 'FileSystems[?Tags[?Key==`fleetsim` && Value==`efs-proof`]].FileSystemId|[0]')
if [ "$FS" = "None" ] || [ -z "$FS" ]; then
  FS=$(aws efs create-file-system --performance-mode generalPurpose --throughput-mode elastic --encrypted \
    --tags Key=fleetsim,Value=efs-proof Key=Name,Value=fleetsim-efs-proof --query 'FileSystemId')
  until [ "$(aws efs describe-file-systems --file-system-id "$FS" --query 'FileSystems[0].LifeCycleState')" = available ]; do sleep 5; done
fi
MT=$(aws efs describe-mount-targets --file-system-id "$FS" --query 'MountTargets[0].MountTargetId')
if [ "$MT" = "None" ] || [ -z "$MT" ]; then
  MT=$(aws efs create-mount-target --file-system-id "$FS" --subnet-id "$DRIVER_SUBNET" \
    --security-groups "$SG_EFS" --query 'MountTargetId')
fi
until [ "$(aws efs describe-mount-targets --mount-target-id "$MT" --query 'MountTargets[0].LifeCycleState')" = available ]; do sleep 5; done
EFS_IP=$(aws efs describe-mount-targets --mount-target-id "$MT" --query 'MountTargets[0].IpAddress')
# One access point, rooted at /fleet and pinned to uid/gid 10001 — the uid `Dockerfile.agent`
# creates and chowns /mnt/efs to. The pin applies to every caller, which is also why the driver can
# read every shard while running as root without any of this depending on the container user.
AP=$(aws efs describe-access-points --file-system-id "$FS" --query 'AccessPoints[0].AccessPointId')
if [ "$AP" = "None" ] || [ -z "$AP" ]; then
  AP=$(aws efs create-access-point --file-system-id "$FS" --posix-user 'Uid=10001,Gid=10001' \
    --root-directory 'Path=/fleet,CreationInfo={OwnerUid=10001,OwnerGid=10001,Permissions=0755}' \
    --tags Key=fleetsim,Value=efs-proof --query 'AccessPointId')
fi
echo "efs $FS at $EFS_IP via $AP"

say "cluster and logs"
aws ecs create-cluster --cluster-name "$CLUSTER" --tags key=fleetsim,value=efs-proof >/dev/null 2>&1 || true
aws logs create-log-group --log-group-name /fleetsim 2>/dev/null || true
aws logs put-retention-policy --log-group-name /fleetsim --retention-in-days 1 2>/dev/null || true

say "task definitions"
EXEC_ROLE="arn:aws:iam::$ACCOUNT:role/fleetsimTaskExecutionRole"
DRIVER_ROLE="arn:aws:iam::$ACCOUNT:role/fleetsimDriverTaskRole"
REPLICA_ROLE="arn:aws:iam::$ACCOUNT:role/fleetsimReplicaTaskRole"
VOLUME="[{\"name\":\"efs\",\"efsVolumeConfiguration\":{\"fileSystemId\":\"$FS\",\"transitEncryption\":\"ENABLED\",\"authorizationConfig\":{\"accessPointId\":\"$AP\",\"iam\":\"DISABLED\"}}}]"
LOGS='{"logDriver":"awslogs","options":{"awslogs-group":"/fleetsim","awslogs-region":"'"$REGION"'","awslogs-stream-prefix":"fleetsim"}}'

register_driver() {
  cat <<JSON > /tmp/fleetsim-td-driver.json
{"family":"fleetsim-driver","networkMode":"awsvpc","requiresCompatibilities":["FARGATE"],
 "cpu":"2048","memory":"8192","executionRoleArn":"$EXEC_ROLE","taskRoleArn":"$DRIVER_ROLE",
 "volumes":$VOLUME,
 "containerDefinitions":[{"name":"driver","image":"$ECR/fleetsim-driver:latest","essential":true,
   "entryPoint":["/bin/bash","-c"],"command":["mkdir -p /mnt/efs && sleep infinity"],
   "linuxParameters":{"initProcessEnabled":true},
   "mountPoints":[{"sourceVolume":"efs","containerPath":"/mnt/efs","readOnly":false}],
   "environment":[{"name":"AWS_REGION","value":"$REGION"},{"name":"FLEETSIM_CLUSTER","value":"$CLUSTER"},
     {"name":"FLEETSIM_EFS_IP","value":"$EFS_IP"},{"name":"FLEETSIM_SG","value":"$SG_TASKS"}],
   "logConfiguration":$LOGS}]}
JSON
  aws ecs register-task-definition --cli-input-json file:///tmp/fleetsim-td-driver.json --query 'taskDefinition.taskDefinitionArn'
}

# `$1` is the extra agent flags — the held-back replica gets `--max-live-sessions 1`.
register_replica() {
  local family=$1 extra=$2
  local agent="/usr/local/bin/beyond-ai-agent serve --service --gateway-url http://$DRIVER_IP:19000 --model claude-test --listen 0.0.0.0:8080 --metrics-listen 127.0.0.1:9090 --grant-key $GRANT_KEY --seal-key /tmp/seal.key --shard s1=/mnt/efs/s1 --drain-grace 30 $extra"
  local script="ulimit -c 0 2>/dev/null || true; mkdir -p /mnt/efs/s1 && printf '%s' '$SEAL_B64' > /tmp/seal.key && $agent & pid=\$!; trap 'kill -TERM \$pid 2>/dev/null' TERM; rc=0; while kill -0 \$pid 2>/dev/null; do wait \$pid; rc=\$?; done; exit \$rc"
  python3 - "$family" "$script" <<'PY' > /tmp/fleetsim-td-replica.json
import json, os, sys
family, script = sys.argv[1], sys.argv[2]
print(json.dumps({
  "family": family, "networkMode": "awsvpc", "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024", "memory": "4096",
  "executionRoleArn": os.environ["EXEC_ROLE"], "taskRoleArn": os.environ["REPLICA_ROLE"],
  "volumes": json.loads(os.environ["VOLUME"]),
  "containerDefinitions": [
    {"name": "agent", "image": os.environ["ECR"] + "/fleetsim-agent:latest", "essential": True,
     "entryPoint": ["/bin/sh", "-c"], "command": [script], "stopTimeout": 45,
     "linuxParameters": {"initProcessEnabled": True},
     "mountPoints": [{"sourceVolume": "efs", "containerPath": "/mnt/efs", "readOnly": False}],
     "logConfiguration": json.loads(os.environ["LOGS"])},
    # The scrape, republished. The agent refuses to bind its metrics listener anywhere but loopback
    # — it describes every tenant on the replica, and the replica is reachable by tenants — so the
    # only way the driver can read it is a process inside the same network namespace, which is
    # exactly what an `awsvpc` sidecar is. This one forwards and does nothing else.
    {"name": "metrics", "image": os.environ["ECR"] + "/fleetsim-driver:latest", "essential": False,
     "entryPoint": ["socat"], "command": ["TCP-LISTEN:9091,fork,reuseaddr", "TCP:127.0.0.1:9090"],
     "logConfiguration": json.loads(os.environ["LOGS"])},
  ],
}, indent=2))
PY
  aws ecs register-task-definition --cli-input-json file:///tmp/fleetsim-td-replica.json --query 'taskDefinition.taskDefinitionArn'
}
export EXEC_ROLE REPLICA_ROLE VOLUME ECR LOGS

# Extra tags are appended to the one `--tags` flag, never passed as a second one: `aws ecs run-task`
# takes the last occurrence, so a second flag silently *replaces* the first — and the tag it would
# have dropped is `fleetsim=efs-proof`, which is the only thing `down.sh` finds anything by.
run_task() {
  local family=$1 subnet=$2
  shift 2
  aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE --task-definition "$family" \
    --enable-execute-command \
    --network-configuration "awsvpcConfiguration={subnets=[$subnet],securityGroups=[$SG_TASKS],assignPublicIp=ENABLED}" \
    --tags key=fleetsim,value=efs-proof "$@" --query 'tasks[0].taskArn'
}
wait_running() {
  local arn=$1
  for _ in $(seq 1 120); do
    st=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --query 'tasks[0].lastStatus')
    [ "$st" = RUNNING ] && return 0
    [ "$st" = STOPPED ] && {
      aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --query 'tasks[0].stoppedReason' >&2
      return 1
    }
    sleep 5
  done
  return 1
}

# The driver first, and not for tidiness: the replicas carry `--gateway-url` from the moment they
# start, so the driver's address has to exist before their task definition does.
register_driver >/dev/null
say "driver"
DRIVER=$(aws ecs list-tasks --cluster "$CLUSTER" --family fleetsim-driver --desired-status RUNNING --query 'taskArns[0]')
if [ "$DRIVER" = "None" ] || [ -z "$DRIVER" ]; then
  DRIVER=$(run_task fleetsim-driver "$DRIVER_SUBNET" key=role,value=driver)
  wait_running "$DRIVER"
fi
DRIVER_IP=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$DRIVER" \
  --query 'tasks[0].attachments[0].details[?name==`privateIPv4Address`].value|[0]')
echo "driver $DRIVER_IP"

say "replicas"
register_replica fleetsim-replica "" >/dev/null
register_replica fleetsim-replica-capped "--max-live-sessions 1" >/dev/null
STARTED=()
for i in $(seq 1 "$REPLICAS"); do
  arn=$(run_task fleetsim-replica "${SUBNETS[$((i - 1))]}" "key=replica,value=r$i")
  STARTED+=("$arn")
done
CAPPED=$(run_task fleetsim-replica-capped "$DRIVER_SUBNET" key=replica,value=capped)
STARTED+=("$CAPPED")
for a in "${STARTED[@]}"; do wait_running "$a"; done

addr_of() {
  aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$1" \
    --query 'tasks[0].attachments[0].details[?name==`privateIPv4Address`].value|[0]'
}

echo
echo "fleet up. from inside the driver task, run the matrix with:"
echo -n "  fleet-sim matrix --substrate attached --fault-cmd /usr/local/bin/fault.sh"
echo -n " --mock-listen $DRIVER_IP:19000 --shard s1=/mnt/efs/s1"
for i in $(seq 1 "$REPLICAS"); do echo -n " --replica $(addr_of "${STARTED[$((i - 1))]}"):8080"; done
for _ in $(seq 1 "$REPLICAS"); do echo -n " --replica-metrics 9091"; done
echo -n " --capped-replica $(addr_of "$CAPPED"):8080=1"
echo
echo
echo "  aws ecs execute-command --cluster $CLUSTER --task $DRIVER --container driver --interactive --command /bin/bash"
