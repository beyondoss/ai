#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2018,SC2019  # JMESPath and the container scripts are single-quoted on purpose
# Bring the whole demo up on AWS: gateway, agent and sandbox on ECS Fargate, sessions on EFS.
#
#     OPENAI_API_KEY=… ./deploy/demo/up.sh            # or OPENROUTER_API_KEY=…
#
# Plain bash and the AWS CLI. Every resource carries `beyond-ai-demo=<name>` and is looked up before
# it is created, so a second run changes nothing and a run that was interrupted can simply be rerun.
# `down.sh` deletes it all and proves it.
#
# What it makes, and why each piece is there:
#
#   VPC, one public subnet     tasks get public IPs, so no NAT gateway (it would cost more than the demo)
#   Cloud Map namespace        gateway/agent/sandbox.<name>.local: names that survive a task being replaced
#   EFS                        the agent's sessions, sealed per tenant; the sandbox mounts it read-only
#   gateway task               the real gateway (crates/gateway) + a NATS sidecar its allowance set needs;
#                              the ONLY task that references the provider key (SSM -> `secrets`)
#   agent task                 `serve --service`, grants verified, sessions on EFS, https callback listener
#   sandbox task               the reference exec endpoint (v1.1-v1.4) running commands for real
#
# Keys never touch the repo: provider keys are read from the environment and stored only as SSM
# SecureStrings under /beyond-ai-demo/<name>/; the signing keys are generated here, once, and stored
# the same way.
set -euo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODEL=${DEMO_MODEL:-gpt-5-mini}
PROVIDER=openai
KEYVAR=OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${OPENROUTER_API_KEY:-}" ]; then
  PROVIDER=openrouter
  KEYVAR=OPENROUTER_API_KEY
  MODEL=${DEMO_MODEL:-openai/gpt-5-mini}
fi

for tool in aws jq docker openssl cargo curl; do
  command -v "$tool" >/dev/null || { echo "up: $tool is required" >&2; exit 2; }
done
ACCOUNT=$(account)
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
WORK=$(mktemp -d)
chmod 700 "$WORK"
trap 'rm -rf "$WORK"' EXIT
# Counted in a file because most creates happen inside $(...) or a pipe, where a variable would be
# lost; printed to stderr because those same places capture stdout.
: > "$WORK/made"
made() { echo x >> "$WORK/made"; echo "  + $*" >&2; }

# ---------------------------------------------------------------------------------------------
say "keys (SSM SecureString under $SSM)"
# A value goes in on stdin and out through a 0600 file: never on a command line `ps` can see.
put_param() { # name, type; value on stdin
  local name=$1 type=$2 f="$WORK/param" cur
  umask 077
  cat > "$f"
  if cur=$(command aws --region "$REGION" --output text ssm get-parameter --with-decryption --name "$SSM/$name" --query Parameter.Value 2>/dev/null); then
    [ "$cur" = "$(cat "$f")" ] && { rm -f "$f"; return 0; }
    command aws --region "$REGION" ssm put-parameter --name "$SSM/$name" --type "$type" --overwrite --value "file://$f" >/dev/null
    echo "  ~ updated $name"
  else
    command aws --region "$REGION" ssm put-parameter --name "$SSM/$name" --type "$type" --value "file://$f" \
      --tags "Key=$TAG_KEY,Value=$NAME" >/dev/null
    made "parameter $name"
  fi
  rm -f "$f"
}
param_exists() { command aws --region "$REGION" ssm get-parameter --name "$SSM/$1" >/dev/null 2>&1; }

KEY=${!KEYVAR:-}
if [ -n "$KEY" ]; then
  printf '%s' "$KEY" | put_param "$PROVIDER-key" SecureString
elif ! param_exists openai-key && ! param_exists openrouter-key; then
  echo "up: set OPENAI_API_KEY (or OPENROUTER_API_KEY); it is stored in SSM and read by the gateway only" >&2
  exit 2
fi
unset KEY OPENAI_API_KEY OPENROUTER_API_KEY
# Which provider is stored decides which the gateway is given, so a rerun without the variable keeps it.
if param_exists openrouter-key && ! param_exists openai-key; then
  PROVIDER=openrouter
  MODEL=${DEMO_MODEL:-openai/gpt-5-mini}
else
  PROVIDER=openai
fi

# Generated once. The marker goes in last, so an interrupted run regenerates the lot instead of
# keeping half of a key set.
if ! param_exists generated; then
  DEMO=$(demo_bin)
  "$DEMO" keys > "$WORK/keys.json"
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -keyout "$WORK/ca.key" \
    -out "$WORK/ca.pem" -days 30 -subj "/CN=beyond-ai-demo-ca" \
    -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign" 2>/dev/null
  openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes -keyout "$WORK/leaf.key" \
    -out "$WORK/leaf.csr" -subj "/CN=agent.$DNS" 2>/dev/null
  printf 'subjectAltName=DNS:agent.%s\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n' "$DNS" > "$WORK/ext.cnf"
  openssl x509 -req -in "$WORK/leaf.csr" -CA "$WORK/ca.pem" -CAkey "$WORK/ca.key" -CAcreateserial \
    -out "$WORK/leaf.pem" -days 30 -extfile "$WORK/ext.cnf" 2>/dev/null
  jq -r .grant_seed "$WORK/keys.json" | tr -d '\n' | put_param grant-seed SecureString
  jq -r .fleet_secret "$WORK/keys.json" | tr -d '\n' | put_param fleet-secret SecureString
  jq -r .dek "$WORK/keys.json" | tr -d '\n' | put_param dek SecureString
  jq -r .gateway_key "$WORK/keys.json" | tr -d '\n' | put_param gateway-key SecureString
  jq -r .gateway_id_signing_key "$WORK/keys.json" | tr -d '\n' | put_param id-signing-key SecureString
  put_param tls-key SecureString < "$WORK/leaf.key"
  jq -n --slurpfile k "$WORK/keys.json" --rawfile ca "$WORK/ca.pem" --rawfile leaf "$WORK/leaf.pem" \
    '{grant_key_flag: $k[0].grant_key_flag, gateway_signing_pub: $k[0].gateway_signing_pub, ca_pem: $ca, tls_cert: $leaf}' \
    | put_param public String
  echo generated | put_param generated String
  rm -f "$WORK"/keys.json "$WORK"/*.key
fi
PUBLIC=$(ssm_get public)
GRANT_KEY_FLAG=$(jq -r .grant_key_flag <<<"$PUBLIC")
GATEWAY_PUB=$(jq -r .gateway_signing_pub <<<"$PUBLIC")
CA_PEM=$(jq -r .ca_pem <<<"$PUBLIC")
TLS_CERT=$(jq -r .tls_cert <<<"$PUBLIC")
param_arn() { echo "arn:aws:ssm:$REGION:$ACCOUNT:parameter$SSM/$1"; }

# ---------------------------------------------------------------------------------------------
say "images"
ecr_repo() {
  q ecr describe-repositories --repository-names "$1" --query 'repositories[0].repositoryName' 2>/dev/null >/dev/null && return 0
  aws ecr create-repository --repository-name "$1" --tags "Key=$TAG_KEY,Value=$NAME" >/dev/null
  made "ecr repository $1"
}
LOGGED_IN=
ensure_image() { # repo, dockerfile
  local repo=$1 file=$2
  ecr_repo "$repo"
  if [ -z "${REBUILD:-}" ] && aws ecr describe-images --repository-name "$repo" --image-ids imageTag=latest >/dev/null 2>&1; then
    return 0
  fi
  if [ -z "$LOGGED_IN" ]; then
    aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR" >/dev/null
    LOGGED_IN=1
  fi
  echo "  building $repo (first time: several minutes)"
  docker build --platform linux/amd64 -q -f "$ROOT/$file" -t "$ECR/$repo:latest" "$ROOT" >/dev/null
  docker push -q "$ECR/$repo:latest" >/dev/null
  made "image $repo:latest"
  IMAGES_CHANGED=1
}
IMAGES_CHANGED=
ensure_image "$P-gateway" Dockerfile
ensure_image "$P-agent" Dockerfile.agent
ensure_image "$P-sandbox" deploy/demo/Dockerfile.sandbox

# ---------------------------------------------------------------------------------------------
say "network"
VPC=$(q ec2 describe-vpcs --filters "Name=tag:$TAG_KEY,Values=$NAME" --query 'Vpcs[0].VpcId')
if [ -z "$VPC" ]; then
  VPC=$(aws ec2 create-vpc --cidr-block "$CIDR" --tag-specifications "ResourceType=vpc,Tags=[$TAGSPEC,{Key=Name,Value=$P}]" --query Vpc.VpcId)
  made "vpc $VPC"
fi
# Cloud Map's private DNS and the EFS mount helper both need these.
aws ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-support
aws ec2 modify-vpc-attribute --vpc-id "$VPC" --enable-dns-hostnames
FILTER="Name=tag:$TAG_KEY,Values=$NAME"

SUBNET=$(q ec2 describe-subnets --filters "$FILTER" --query 'Subnets[0].SubnetId')
if [ -z "$SUBNET" ]; then
  SUBNET=$(aws ec2 create-subnet --vpc-id "$VPC" --cidr-block "$SUBNET_CIDR" --availability-zone "$AZ" \
    --tag-specifications "ResourceType=subnet,Tags=[$TAGSPEC,{Key=Name,Value=$P}]" --query Subnet.SubnetId)
  made "subnet $SUBNET"
fi

IGW=$(q ec2 describe-internet-gateways --filters "$FILTER" --query 'InternetGateways[0].InternetGatewayId')
if [ -z "$IGW" ]; then
  IGW=$(aws ec2 create-internet-gateway --tag-specifications "ResourceType=internet-gateway,Tags=[$TAGSPEC]" \
    --query InternetGateway.InternetGatewayId)
  made "internet gateway $IGW"
fi
attached=$(q ec2 describe-internet-gateways --internet-gateway-ids "$IGW" --query 'InternetGateways[0].Attachments[0].VpcId')
[ "$attached" = "$VPC" ] || aws ec2 attach-internet-gateway --internet-gateway-id "$IGW" --vpc-id "$VPC"

RT=$(q ec2 describe-route-tables --filters "$FILTER" --query 'RouteTables[0].RouteTableId')
if [ -z "$RT" ]; then
  RT=$(aws ec2 create-route-table --vpc-id "$VPC" --tag-specifications "ResourceType=route-table,Tags=[$TAGSPEC]" \
    --query RouteTable.RouteTableId)
  made "route table $RT"
fi
[ -n "$(q ec2 describe-route-tables --route-table-ids "$RT" --query 'RouteTables[0].Routes[?GatewayId==`'"$IGW"'`].GatewayId|[0]')" ] \
  || aws ec2 create-route --route-table-id "$RT" --destination-cidr-block 0.0.0.0/0 --gateway-id "$IGW" >/dev/null
[ -n "$(q ec2 describe-route-tables --route-table-ids "$RT" --query 'RouteTables[0].Associations[?SubnetId==`'"$SUBNET"'`].SubnetId|[0]')" ] \
  || aws ec2 associate-route-table --route-table-id "$RT" --subnet-id "$SUBNET" >/dev/null

sg_for() { # name, description -> id
  local id
  id=$(q ec2 describe-security-groups --filters "$FILTER" "Name=group-name,Values=$1" --query 'SecurityGroups[0].GroupId')
  if [ -z "$id" ]; then
    id=$(aws ec2 create-security-group --group-name "$1" --description "$2" --vpc-id "$VPC" \
      --tag-specifications "ResourceType=security-group,Tags=[$TAGSPEC]" --query GroupId)
    made "security group $1"
  fi
  echo "$id"
}
SG_TASKS=$(sg_for "$P-tasks" "beyond-ai-demo tasks")
SG_EFS=$(sg_for "$P-efs" "beyond-ai-demo efs mount target")
# Tasks talk to each other freely; the only way in from outside is the agent's websocket port, from
# the caller's address alone.
sg_rule() { # group, extra authorize args...; counts only a rule that was new
  local out
  if out=$(command aws --region "$REGION" ec2 authorize-security-group-ingress --group-id "$1" "${@:2}" 2>&1); then
    made "ingress rule on $1"
  elif [[ $out != *InvalidPermission.Duplicate* ]]; then
    echo "$out" >&2
    return 1
  fi
}
sg_rule "$SG_TASKS" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_TASKS}]"
sg_rule "$SG_EFS" --protocol tcp --port 2049 --source-group "$SG_TASKS"
CALLER=${DEMO_ALLOW_CIDR:-}
if [ -z "$CALLER" ]; then
  CALLER="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32"
fi
sg_rule "$SG_TASKS" --protocol tcp --port "$AGENT_PORT" --cidr "$CALLER"

# ---------------------------------------------------------------------------------------------
say "efs"
FS=$(q efs describe-file-systems --creation-token "$P" --query 'FileSystems[0].FileSystemId')
if [ -z "$FS" ]; then
  FS=$(aws efs create-file-system --creation-token "$P" --encrypted \
    --tags "Key=$TAG_KEY,Value=$NAME" "Key=Name,Value=$P" --query FileSystemId)
  made "file system $FS"
fi
until [ "$(aws efs describe-file-systems --file-system-id "$FS" --query 'FileSystems[0].LifeCycleState')" = available ]; do sleep 3; done
MT=$(q efs describe-mount-targets --file-system-id "$FS" --query 'MountTargets[0].MountTargetId')
if [ -z "$MT" ]; then
  MT=$(aws efs create-mount-target --file-system-id "$FS" --subnet-id "$SUBNET" --security-groups "$SG_EFS" --query MountTargetId)
  made "mount target $MT"
fi
# The agent runs as uid 10001 (Dockerfile.agent); the access point pins every caller to it, the
# sandbox included, so what the sandbox reads is what the agent could.
AP=$(q efs describe-access-points --file-system-id "$FS" --query 'AccessPoints[0].AccessPointId')
if [ -z "$AP" ]; then
  AP=$(aws efs create-access-point --file-system-id "$FS" --posix-user Uid=10001,Gid=10001 \
    --root-directory 'Path=/demo,CreationInfo={OwnerUid=10001,OwnerGid=10001,Permissions=0755}' \
    --tags "Key=$TAG_KEY,Value=$NAME" "Key=Name,Value=$P" --query AccessPointId)
  made "access point $AP"
fi
until [ "$(aws efs describe-mount-targets --mount-target-id "$MT" --query 'MountTargets[0].LifeCycleState')" = available ]; do sleep 3; done
until [ "$(aws efs describe-access-points --access-point-id "$AP" --query 'AccessPoints[0].LifeCycleState')" = available ]; do sleep 2; done

# ---------------------------------------------------------------------------------------------
say "iam"
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
role() { # suffix, space-separated parameter names it may read (empty: none)
  local r="$P-$1" fresh=
  if ! aws iam get-role --role-name "$r" >/dev/null 2>&1; then
    aws iam create-role --role-name "$r" --assume-role-policy-document "$TRUST" --tags "Key=$TAG_KEY,Value=$NAME" >/dev/null
    made "role $r"
    fresh=1
  fi
  aws iam attach-role-policy --role-name "$r" --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  if [ -n "$2" ]; then
    local arns
    arns=$(for n in $2; do param_arn "$n"; done | jq -R . | jq -s .)
    aws iam put-role-policy --role-name "$r" --policy-name read-own-parameters --policy-document \
      "$(jq -n --argjson r "$arns" '{Version:"2012-10-17",Statement:[{Effect:"Allow",Action:["ssm:GetParameters"],Resource:$r}]}')"
  fi
  [ -z "$fresh" ] || sleep 10   # a new role is not assumable by ECS for a few seconds
}
# The gateway's role reads the provider key and its id-signing key and nothing else; the agent's
# reads its own seal and TLS keys; the sandbox's reads nothing. No role can read the whole prefix.
role gateway-exec "$PROVIDER-key id-signing-key"
role agent-exec "fleet-secret tls-key"
role sandbox-exec ""
EXEC() { echo "arn:aws:iam::$ACCOUNT:role/$P-$1-exec"; }

# ---------------------------------------------------------------------------------------------
say "cluster, logs, names"
if [ -z "$(q ecs describe-clusters --clusters "$CLUSTER" --query 'clusters[?status==`ACTIVE`].clusterName|[0]')" ]; then
  aws ecs create-cluster --cluster-name "$CLUSTER" --tags "key=$TAG_KEY,value=$NAME" >/dev/null
  made "cluster $CLUSTER"
fi
if [ -z "$(q logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName|[0]")" ]; then
  aws logs create-log-group --log-group-name "$LOG_GROUP" --tags "$TAG_KEY=$NAME"
  made "log group $LOG_GROUP"
fi
aws logs put-retention-policy --log-group-name "$LOG_GROUP" --retention-in-days 1

NSID=$(q servicediscovery list-namespaces --query "Namespaces[?Name=='$DNS'].Id|[0]")
if [ -z "$NSID" ]; then
  op=$(aws servicediscovery create-private-dns-namespace --name "$DNS" --vpc "$VPC" \
    --tags "Key=$TAG_KEY,Value=$NAME" --query OperationId 2>/dev/null || true)
  for _ in $(seq 1 60); do
    NSID=$(q servicediscovery list-namespaces --query "Namespaces[?Name=='$DNS'].Id|[0]")
    [ -n "$NSID" ] && break
    sleep 3
  done
  [ -n "$NSID" ] || { echo "up: the namespace $DNS did not appear ($op)" >&2; exit 1; }
  made "namespace $DNS"
fi
registry() { # service name -> registry ARN
  local arn
  arn=$(q servicediscovery list-services --filters "Name=NAMESPACE_ID,Values=$NSID,Condition=EQ" --query "Services[?Name=='$1'].Arn|[0]")
  if [ -z "$arn" ]; then
    arn=$(aws servicediscovery create-service --name "$1" --namespace-id "$NSID" \
      --dns-config 'RoutingPolicy=MULTIVALUE,DnsRecords=[{Type=A,TTL=10}]' \
      --tags "Key=$TAG_KEY,Value=$NAME" --query Service.Arn)
    made "name $1.$DNS"
  fi
  echo "$arn"
}

# ---------------------------------------------------------------------------------------------
say "task definitions"
LOGS() { jq -n --arg g "$LOG_GROUP" --arg r "$REGION" --arg p "$1" \
  '{logDriver:"awslogs",options:{"awslogs-group":$g,"awslogs-region":$r,"awslogs-stream-prefix":$p}}'; }
EFSVOL=$(jq -n --arg fs "$FS" --arg ap "$AP" \
  '[{name:"efs",efsVolumeConfiguration:{fileSystemId:$fs,transitEncryption:"ENABLED",authorizationConfig:{accessPointId:$ap,iam:"DISABLED"}}}]')

# Registered only when the definition changed (its hash is a tag on the revision), so a rerun does
# not mint a new revision.
register() { # family, json (stdin); echoes the ARN in use
  local family=$1 spec hash cur
  spec=$(cat)
  hash=$(printf '%s' "$spec" | sha256sum | cut -c1-16)
  cur=$(q ecs describe-task-definition --task-definition "$family" --include TAGS \
    --query 'tags[?key==`spec`].value|[0]' 2>/dev/null || true)
  if [ "$cur" = "$hash" ]; then
    q ecs describe-task-definition --task-definition "$family" --query 'taskDefinition.taskDefinitionArn'
    return 0
  fi
  printf '%s' "$spec" > "$WORK/td.json"
  made "task definition $family"
  aws ecs register-task-definition --cli-input-json "file://$WORK/td.json" \
    --tags "key=spec,value=$hash" "key=$TAG_KEY,value=$NAME" --query 'taskDefinition.taskDefinitionArn'
}

TD_GATEWAY=$(jq -n --arg family "$P-gateway" --arg exec "$(EXEC gateway)" --argjson logs_gw "$(LOGS gateway)" \
  --argjson logs_nats "$(LOGS nats)" --arg img "$ECR/$P-gateway:latest" --arg pub "$GATEWAY_PUB" \
  --arg pk "$(param_arn "$PROVIDER-key")" --arg ik "$(param_arn id-signing-key)" \
  --arg poolvar "AI_POOL_KEY_$(tr a-z A-Z <<<"$PROVIDER")" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"512", memory:"1024",
  executionRoleArn:$exec,
  containerDefinitions:[
    {name:"nats", image:"public.ecr.aws/docker/library/nats:2.10-alpine", essential:true,
     command:["-js","-a","127.0.0.1"], logConfiguration:$logs_nats},
    {name:"gateway", image:$img, essential:true, dependsOn:[{containerName:"nats",condition:"START"}],
     portMappings:[{containerPort:8080}],
     environment:[{name:"AI_NATS_URL",value:"nats://127.0.0.1:4222"},{name:"AI_SIGNING_KEY_1",value:$pub}],
     secrets:[{name:$poolvar,valueFrom:$pk},{name:"AI_ID_SIGNING_KEY_1",valueFrom:$ik}],
     logConfiguration:$logs_gw}]}' | register "$P-gateway")

# The agent is NOT pid 1: pid 1 is a shell that never forwards SIGTERM. A stopped task is therefore
# SIGKILLed two seconds later, agent included, with no chance to seal its segment or release its
# lock: the hard kill `demo.sh kill` is about, done with nothing but `aws ecs stop-task`.
AGENT_SCRIPT='printf "%s" "$TLS_CERT" > /tmp/leaf.pem; printf "%s" "$TLS_KEY" > /tmp/leaf.key; ulimit -c 0 2>/dev/null || true
/usr/local/bin/beyond-ai-agent serve --service --listen 0.0.0.0:8080 --tls-listen 0.0.0.0:8443 \
  --tls-cert /tmp/leaf.pem --tls-key /tmp/leaf.key --gateway-url "$GATEWAY_URL" --model "$MODEL" \
  --notify-origin "$NOTIFY_ORIGIN" --grant-key "$GRANT_KEY_FLAG" --shard demo=/mnt/efs &
wait $!'
TD_AGENT=$(jq -n --arg family "$P-agent" --arg exec "$(EXEC agent)" --argjson logs "$(LOGS agent)" \
  --arg img "$ECR/$P-agent:latest" --argjson vol "$EFSVOL" --arg script "$AGENT_SCRIPT" \
  --arg gw "http://gateway.$DNS:$GATEWAY_PORT" --arg model "$MODEL" --arg origin "https://agent.$DNS:$AGENT_TLS_PORT" \
  --arg grant "$GRANT_KEY_FLAG" --arg cert "$TLS_CERT" --arg seal "$(param_arn fleet-secret)" --arg tlskey "$(param_arn tls-key)" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"512", memory:"1024",
  executionRoleArn:$exec, volumes:$vol,
  containerDefinitions:[
    {name:"agent", image:$img, essential:true, stopTimeout:2, entryPoint:["/bin/sh","-c"], command:[$script],
     portMappings:[{containerPort:8080},{containerPort:8443}],
     mountPoints:[{sourceVolume:"efs",containerPath:"/mnt/efs",readOnly:false}],
     environment:[{name:"HOME",value:"/tmp"},{name:"GATEWAY_URL",value:$gw},{name:"MODEL",value:$model},
       {name:"NOTIFY_ORIGIN",value:$origin},{name:"GRANT_KEY_FLAG",value:$grant},{name:"TLS_CERT",value:$cert}],
     secrets:[{name:"AI_AGENT_SEAL_KEY_B64",valueFrom:$seal},{name:"TLS_KEY",valueFrom:$tlskey}],
     logConfiguration:$logs}]}' | register "$P-agent")

SANDBOX_SCRIPT='printf "%s" "$CA_PEM" > /tmp/ca.pem
exec /usr/local/bin/beyond-ai-demo exec-stub --listen 0.0.0.0:8081 --root /workspace --home /home/sandbox'
TD_SANDBOX=$(jq -n --arg family "$P-sandbox" --arg exec "$(EXEC sandbox)" --argjson logs "$(LOGS sandbox)" \
  --arg img "$ECR/$P-sandbox:latest" --argjson vol "$EFSVOL" --arg script "$SANDBOX_SCRIPT" --arg ca "$CA_PEM" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"256", memory:"512",
  executionRoleArn:$exec, volumes:$vol,
  containerDefinitions:[
    {name:"sandbox", image:$img, essential:true, entryPoint:["/bin/sh","-c"], command:[$script],
     portMappings:[{containerPort:8081}],
     mountPoints:[{sourceVolume:"efs",containerPath:"/mnt/efs",readOnly:true}],
     environment:[{name:"CA_PEM",value:$ca},{name:"SSL_CERT_FILE",value:"/tmp/ca.pem"}],
     logConfiguration:$logs}]}' | register "$P-sandbox")

# ---------------------------------------------------------------------------------------------
say "services"
service() { # name, task definition arn, port
  local name="$P-$1" td=$2 cur
  local reg
  reg=$(registry "$1")
  cur=$(q ecs describe-services --cluster "$CLUSTER" --services "$name" --query 'services[?status==`ACTIVE`].taskDefinition|[0]')
  if [ -z "$cur" ]; then
    aws ecs create-service --cluster "$CLUSTER" --service-name "$name" --task-definition "$td" --desired-count 1 \
      --launch-type FARGATE --platform-version LATEST \
      --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$SG_TASKS],assignPublicIp=ENABLED}" \
      --service-registries "registryArn=$reg" --propagate-tags SERVICE --enable-ecs-managed-tags \
      --tags "key=$TAG_KEY,value=$NAME" >/dev/null
    made "service $name"
  elif [ "$cur" != "$td" ] || [ -n "$IMAGES_CHANGED" ]; then
    aws ecs update-service --cluster "$CLUSTER" --service "$name" --task-definition "$td" --desired-count 1 \
      ${IMAGES_CHANGED:+--force-new-deployment} >/dev/null
    echo "  ~ redeploying $name"
  fi
}
# An ECS service tag cannot be added after create when the service already has none; fine here, ours
# are tagged at create.
service gateway "$TD_GATEWAY"
service agent "$TD_AGENT"
service sandbox "$TD_SANDBOX"

echo "waiting for the three services to be stable..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "$P-gateway" "$P-agent" "$P-sandbox"

IP=$(agent_ip)
echo
created=$(wc -l < "$WORK/made" 2>/dev/null || echo 0)
echo "up: created $created resource(s) in $REGION (tag $TAG_KEY=$NAME)."
echo "  agent   ws://$IP:$AGENT_PORT   (open to $CALLER only)"
echo "  model   $MODEL via the gateway, provider $PROVIDER"
echo "  next    ./deploy/demo/demo.sh        a scripted conversation, with a run record and report"
echo "          ./deploy/demo/demo.sh kill   kill the agent mid-job; the session survives"
echo "          ./deploy/demo/down.sh        delete everything, then prove it is gone"
