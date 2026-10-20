#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2018,SC2019  # JMESPath and the container scripts are single-quoted on purpose
# Delete everything the demo made, in dependency order, tolerating whatever is already gone, and then
# VERIFY: the tagging API and an explicit listing per service for every type `up.sh` creates (several
# of which the tagging API misses or reports late). Exits non-zero, naming what remains, if anything
# does. Safe to run twice, and safe to run after an `up.sh` or a `down.sh` that was interrupted.
#
#     ./deploy/demo/down.sh
#
# The result (a count per resource type) is merged into the latest run record under
# results/ as `teardown`, and its report is rebuilt, so the page shows every type ticking to 0.
set -uo pipefail
# shellcheck source=lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

FILTER="Name=tag:$TAG_KEY,Values=$NAME"
# E2B sandboxes are part of the deployment whenever E2B_API_KEY is there to look for them. In e2b
# mode a missing key is an error: "nothing left" could not be proven.
if [ -z "${E2B_API_KEY:-}" ] && [ "$USE_E2B" = 1 ]; then
  echo "down: SANDBOX=e2b or both needs E2B_API_KEY to find and kill the sandboxes" >&2
  exit 2
fi
quiet() { "$@" >/dev/null 2>&1; }
# Retry a delete that fails while something it depends on is still going away.
retry() { # attempts, command...
  local n=$1 i
  shift
  for ((i = 0; i < n; i++)); do
    "$@" >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

# What there is, per resource type. Taken before the deletes (so the report can count down from it)
# and again after them (the gate).
declare -A LEFT
n() { local v; v=$(eval "$1" 2>/dev/null) || v=0; [ -z "$v" ] || [ "$v" = None ] && v=0; echo "$v"; }
# Revision ARNs of this demo's task definitions in one status. `list-task-definitions` filters on a
# FULL family name, not a prefix, so ask the families first.
td_arns() { # status
  local fam
  for fam in $(aws ecs list-task-definition-families --family-prefix "$P-" --status ALL --query 'families' 2>/dev/null); do
    aws ecs list-task-definitions --family-prefix "$fam" --status "$1" --query 'taskDefinitionArns' 2>/dev/null
  done | tr '\t' '\n' | grep -c . || true
}
survey() {
  VPC_NOW=$(q ec2 describe-vpcs --filters "$FILTER" --query 'Vpcs[0].VpcId' 2>/dev/null || true)
  NS_NOW=$(q servicediscovery list-namespaces --query "Namespaces[?Name=='$DNS'].Id|[0]" 2>/dev/null || true)
  LEFT[ssm_parameters]=$(n "aws ssm get-parameters-by-path --path '$SSM' --recursive --query 'length(Parameters)'")
  LEFT[iam_roles]=$(n "aws iam list-roles --query \"length(Roles[?starts_with(RoleName,'$P-')])\"")
  LEFT[log_groups]=$(n "aws logs describe-log-groups --log-group-name-prefix '$LOG_GROUP' --query 'length(logGroups)'")
  LEFT[ecs_clusters]=$(n "aws ecs describe-clusters --clusters '$CLUSTER' --query 'length(clusters[?status!=\`INACTIVE\`])'")
  LEFT[ecs_services]=$(n "aws ecs list-services --cluster '$CLUSTER' --query 'length(serviceArns)'")
  LEFT[ecs_tasks]=$(n "aws ecs list-tasks --cluster '$CLUSTER' --query 'length(taskArns)'")
  LEFT[ecs_task_definitions]=$(( $(td_arns ACTIVE) + $(td_arns INACTIVE) ))
  LEFT[cloudmap_namespaces]=$(n "aws servicediscovery list-namespaces --query \"length(Namespaces[?Name=='$DNS'])\"")
  if [ -n "$NS_NOW" ]; then
    LEFT[cloudmap_services]=$(n "aws servicediscovery list-services --filters 'Name=NAMESPACE_ID,Values=$NS_NOW,Condition=EQ' --query 'length(Services)'")
  else
    LEFT[cloudmap_services]=0
  fi
  LEFT[route53_hosted_zones]=$(n "aws route53 list-hosted-zones --query \"length(HostedZones[?Name=='$DNS.'])\"")
  LEFT[efs_file_systems]=$(n "aws efs describe-file-systems --creation-token '$P' --query 'length(FileSystems)'")
  LEFT[efs_mount_targets]=$(n "aws efs describe-file-systems --creation-token '$P' --query 'FileSystems[0].NumberOfMountTargets'")
  LEFT[efs_access_points]=$(n "aws efs describe-access-points --query \"length(AccessPoints[?Tags[?Key=='$TAG_KEY' && Value=='$NAME']])\"")
  LEFT[ec2_vpcs]=$(n "aws ec2 describe-vpcs --filters '$FILTER' --query 'length(Vpcs)'")
  LEFT[ec2_subnets]=$(n "aws ec2 describe-subnets --filters '$FILTER' --query 'length(Subnets)'")
  LEFT[ec2_internet_gateways]=$(n "aws ec2 describe-internet-gateways --filters '$FILTER' --query 'length(InternetGateways)'")
  LEFT[ec2_route_tables]=$(n "aws ec2 describe-route-tables --filters '$FILTER' --query 'length(RouteTables)'")
  LEFT[ec2_security_groups]=$(n "aws ec2 describe-security-groups --filters '$FILTER' --query 'length(SecurityGroups)'")
  if [ -n "$VPC_NOW" ]; then
    LEFT[ec2_network_interfaces]=$(n "aws ec2 describe-network-interfaces --filters 'Name=vpc-id,Values=$VPC_NOW' --query 'length(NetworkInterfaces)'")
  else
    LEFT[ec2_network_interfaces]=0
  fi
  if [ -n "${E2B_API_KEY:-}" ]; then
    # A listing that cannot be read is not a zero.
    LEFT[e2b_sandboxes]=$(e2b_count 2>/dev/null) || { echo "down: could not list E2B sandboxes" >&2; LEFT[e2b_sandboxes]=1; }
  fi
  LEFT[ecr_repositories]=$(n "aws ecr describe-repositories --query \"length(repositories[?starts_with(repositoryName,'$P-')])\"")
}
counts_json() {
  local json='{}' k
  for k in $(printf '%s\n' "${!LEFT[@]}" | sort); do
    json=$(jq -c --arg k "$k" --argjson v "${LEFT[$k]}" '. + {($k): $v}' <<<"$json")
  done
  echo "$json"
}

survey
BEFORE=$(counts_json)

# ------------------------------------------------------------------------------------------------
if [ -n "${E2B_API_KEY:-}" ]; then
  say "e2b sandboxes"
  for _ in 1 2 3; do
    ids=$(e2b_ids 2>/dev/null) || ids=""
    [ -n "$ids" ] || break
    for id in $ids; do
      echo "killing E2B sandbox $id (HTTP $(e2b_kill "$id"))"
    done
    sleep 3
  done
fi

say "ecs services and tasks"
if [ -n "$(q ecs describe-clusters --clusters "$CLUSTER" --query 'clusters[?status==`ACTIVE`].clusterName|[0]')" ]; then
  for svc in $(aws ecs list-services --cluster "$CLUSTER" --query 'serviceArns' 2>/dev/null); do
    aws ecs update-service --cluster "$CLUSTER" --service "$svc" --desired-count 0 >/dev/null 2>&1
    aws ecs delete-service --cluster "$CLUSTER" --service "$svc" --force >/dev/null 2>&1 && echo "deleting $svc"
  done
  for t in $(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns' 2>/dev/null); do
    aws ecs stop-task --cluster "$CLUSTER" --task "$t" >/dev/null 2>&1 && echo "stopping $t"
  done
  for _ in $(seq 1 60); do
    [ "$(aws ecs list-tasks --cluster "$CLUSTER" --query 'length(taskArns)' 2>/dev/null)" = 0 ] && break
    sleep 5
  done
  retry 24 aws ecs delete-cluster --cluster "$CLUSTER" && echo "deleted cluster $CLUSTER"
fi

say "cloud map"
NSID=$(q servicediscovery list-namespaces --query "Namespaces[?Name=='$DNS'].Id|[0]" 2>/dev/null || true)
if [ -n "$NSID" ]; then
  for s in $(aws servicediscovery list-services --filters "Name=NAMESPACE_ID,Values=$NSID,Condition=EQ" --query 'Services[].Id' 2>/dev/null); do
    for i in $(aws servicediscovery list-instances --service-id "$s" --query 'Instances[].Id' 2>/dev/null); do
      quiet aws servicediscovery deregister-instance --service-id "$s" --instance-id "$i"
    done
    # Instances deregister asynchronously, and a service with one left refuses to go.
    retry 24 aws servicediscovery delete-service --id "$s" && echo "deleted name $s"
  done
  retry 24 aws servicediscovery delete-namespace --id "$NSID" && echo "deleted namespace $DNS"
  for _ in $(seq 1 30); do
    [ -z "$(q servicediscovery list-namespaces --query "Namespaces[?Name=='$DNS'].Id|[0]")" ] && break
    sleep 5
  done
fi

say "efs"
# Access points and mount targets before the file system; EFS refuses a file system that has either.
for fs in $(aws efs describe-file-systems --query 'FileSystems[?Tags[?Key==`'"$TAG_KEY"'` && Value==`'"$NAME"'`]].FileSystemId' 2>/dev/null); do
  for ap in $(aws efs describe-access-points --file-system-id "$fs" --query 'AccessPoints[].AccessPointId' 2>/dev/null); do
    quiet aws efs delete-access-point --access-point-id "$ap" && echo "deleted $ap"
  done
  for mt in $(aws efs describe-mount-targets --file-system-id "$fs" --query 'MountTargets[].MountTargetId' 2>/dev/null); do
    quiet aws efs delete-mount-target --mount-target-id "$mt" && echo "deleted $mt"
  done
  for _ in $(seq 1 60); do
    [ "$(aws efs describe-mount-targets --file-system-id "$fs" --query 'length(MountTargets)' 2>/dev/null)" = 0 ] \
      && [ "$(aws efs describe-access-points --file-system-id "$fs" --query 'length(AccessPoints)' 2>/dev/null)" = 0 ] && break
    sleep 5
  done
  retry 24 aws efs delete-file-system --file-system-id "$fs" && echo "deleted $fs"
done

say "network"
VPC=$(q ec2 describe-vpcs --filters "$FILTER" --query 'Vpcs[0].VpcId' 2>/dev/null || true)
if [ -n "$VPC" ]; then
  # A stopped task's ENI lingers a minute or two and holds the subnet and the group.
  for _ in $(seq 1 90); do
    [ "$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$VPC" --query 'length(NetworkInterfaces)' 2>/dev/null)" = 0 ] && break
    sleep 5
  done
  for sub in $(aws ec2 describe-subnets --filters "$FILTER" --query 'Subnets[].SubnetId' 2>/dev/null); do
    retry 12 aws ec2 delete-subnet --subnet-id "$sub" && echo "deleted $sub"
  done
  # The two groups reference each other, so a pass can fail on whichever the listing returns first.
  for _ in 1 2 3; do
    for sg in $(aws ec2 describe-security-groups --filters "$FILTER" --query 'SecurityGroups[].GroupId' 2>/dev/null); do
      quiet aws ec2 delete-security-group --group-id "$sg" && echo "deleted $sg"
    done
  done
  for igw in $(aws ec2 describe-internet-gateways --filters "$FILTER" --query 'InternetGateways[].InternetGatewayId' 2>/dev/null); do
    quiet aws ec2 detach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$VPC"
    quiet aws ec2 delete-internet-gateway --internet-gateway-id "$igw" && echo "deleted $igw"
  done
  for rt in $(aws ec2 describe-route-tables --filters "$FILTER" --query 'RouteTables[].RouteTableId' 2>/dev/null); do
    quiet aws ec2 delete-route-table --route-table-id "$rt" && echo "deleted $rt"
  done
  retry 12 aws ec2 delete-vpc --vpc-id "$VPC" && echo "deleted $VPC"
fi

say "task definitions, logs, images, roles, parameters"
tds=
for fam in $(aws ecs list-task-definition-families --family-prefix "$P-" --status ALL --query 'families' 2>/dev/null); do
  for td in $(aws ecs list-task-definitions --family-prefix "$fam" --status ACTIVE --query 'taskDefinitionArns' 2>/dev/null); do
    quiet aws ecs deregister-task-definition --task-definition "$td"
  done
  tds="$tds $(aws ecs list-task-definitions --family-prefix "$fam" --status INACTIVE --query 'taskDefinitionArns' 2>/dev/null)"
done
# delete-task-definitions takes ten at a time.
# shellcheck disable=SC2086
set -- $tds
while [ $# -gt 0 ]; do
  batch=("${@:1:10}")
  quiet aws ecs delete-task-definitions --task-definitions "${batch[@]}" && echo "deleted ${#batch[@]} task definition revision(s)"
  shift $((${#batch[@]}))
done
quiet aws logs delete-log-group --log-group-name "$LOG_GROUP" && echo "deleted log group"
for repo in $(aws ecr describe-repositories --query "repositories[?starts_with(repositoryName,'$P-')].repositoryName" 2>/dev/null); do
  quiet aws ecr delete-repository --repository-name "$repo" --force && echo "deleted repository $repo"
done
for role in $(aws iam list-roles --query "Roles[?starts_with(RoleName,'$P-')].RoleName" 2>/dev/null); do
  for pol in $(aws iam list-role-policies --role-name "$role" --query 'PolicyNames' 2>/dev/null); do
    quiet aws iam delete-role-policy --role-name "$role" --policy-name "$pol"
  done
  for arn in $(aws iam list-attached-role-policies --role-name "$role" --query 'AttachedPolicies[].PolicyArn' 2>/dev/null); do
    quiet aws iam detach-role-policy --role-name "$role" --policy-arn "$arn"
  done
  quiet aws iam delete-role --role-name "$role" && echo "deleted role $role"
done
params=$(aws ssm get-parameters-by-path --path "$SSM" --recursive --query 'Parameters[].Name' 2>/dev/null)
# shellcheck disable=SC2086
set -- $params
while [ $# -gt 0 ]; do
  batch=("${@:1:10}")
  quiet aws ssm delete-parameters --names "${batch[@]}" && echo "deleted ${#batch[@]} parameter(s)"
  shift $((${#batch[@]}))
done

# ------------------------------------------------------------------------------------------------
# The gate. "I ran the teardown" and "nothing is left" are different claims; only one is checkable.
say "what is left"
survey
# The tagging API last, and waited on: it reports a deleted resource for a while after it is gone.
# ECS keeps a deleted cluster or service and every stopped task in the tagging index for hours, so an
# entry only counts if the service's own API cannot show it gone. What was only index lag is reported.
tagged_live() {
  local arn st cl
  for arn in $(aws resourcegroupstaggingapi get-resources --tag-filters "Key=$TAG_KEY,Values=$NAME" --query 'ResourceTagMappingList[].ResourceARN' 2>/dev/null); do
    case "$arn" in
      *:ecs:*:cluster/*)
        st=$(aws ecs describe-clusters --clusters "$arn" --query 'clusters[0].status' 2>/dev/null)
        [ "$st" = ACTIVE ] || [ "$st" = PROVISIONING ] || [ "$st" = DEPROVISIONING ] && echo "$arn" ;;
      *:ecs:*:service/*)
        cl=${arn#*:service/}; cl=${cl%%/*}
        st=$(aws ecs describe-services --cluster "$cl" --services "$arn" --query 'services[0].status' 2>/dev/null)
        [ "$st" = ACTIVE ] || [ "$st" = DRAINING ] && echo "$arn" ;;
      *:ecs:*:task/*)
        cl=${arn#*:task/}; cl=${cl%%/*}
        st=$(aws ecs describe-tasks --cluster "$cl" --tasks "$arn" --query 'tasks[0].lastStatus' 2>/dev/null)
        [ -n "$st" ] && [ "$st" != None ] && [ "$st" != STOPPED ] && echo "$arn" ;;
      *:ecs:*:task-definition/*)
        st=$(aws ecs describe-task-definition --task-definition "$arn" --query 'taskDefinition.status' 2>/dev/null)
        [ "$st" = ACTIVE ] || [ "$st" = INACTIVE ] && echo "$arn" ;;
      *) echo "$arn" ;;
    esac
  done | grep -c . || true
}
tagged_all() { n "aws resourcegroupstaggingapi get-resources --tag-filters 'Key=$TAG_KEY,Values=$NAME' --query 'length(ResourceTagMappingList)'"; }
TAG_INDEX=$(tagged_all)
LEFT[tagging_api]=$(tagged_live)

total=0
for k in $(printf '%s\n' "${!LEFT[@]}" | sort); do
  printf '  %-24s %s   (was %s)\n' "$k" "${LEFT[$k]}" "$(jq -r --arg k "$k" '.[$k] // 0' <<<"$BEFORE")"
  [ "$k" = tagging_api ] && [ "$TAG_INDEX" != 0 ] && printf '  %-24s %s listed by the index, none of them still existing (ECS index lag)\n' "" "$TAG_INDEX"
  total=$((total + ${LEFT[$k]}))
done
json=$(counts_json)
verdict=$([ "$total" = 0 ] && echo clean || echo remains)
record=$(jq -nc --argjson counts "$json" --argjson before "$BEFORE" --arg v "$verdict" --arg name "$NAME" --arg region "$REGION" \
  --argjson stale "$TAG_INDEX" --arg at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '{tagging_index_stale:$stale,name:$name,region:$region,verified_at:$at,verdict:$v,before:$before,counts:$counts}')

# Into the run records still waiting for it (every use case's run since the last teardown), so each
# report's teardown panel has it; with none waiting, into the latest one.
RESULTS=${DEMO_RESULTS:-$HERE/results}
records=$(find "$RESULTS" -mindepth 2 -maxdepth 2 -name run.json 2>/dev/null | sort)
targets=
for r in $records; do
  [ "$(jq -r '.teardown == null' "$r")" = true ] && targets="$targets $r"
done
if [ -z "$targets" ]; then
  latest=$(printf '%s\n' "$records" | tail -1)
  # A second run over an already clean record would overwrite what the first one counted down from.
  if [ -n "$latest" ] && { [ "$(jq -r '.teardown.verdict // ""' "$latest")" != clean ] || [ "$(jq '[.[]]|add' <<<"$BEFORE")" != 0 ]; }; then
    targets=$latest
  fi
fi
if [ "$verdict" = clean ]; then
  for rec in $targets; do
    jq --argjson t "$record" '.teardown = $t
      | .checks |= map(if .name == "teardown-zero" then
          .pass = true | .status = "pass" | .started_ms = null | .ended_ms = null
          | .detail = ("every resource type counted 0 at " + $t.verified_at)
          | .headline = {label: "left behind", value: "0", unit: "resources"}
          | .measurements = {resource_types: ($t.counts | length), remaining: ($t.counts | add)}
        else . end)' "$rec" >"$rec.tmp" && mv "$rec.tmp" "$rec"
    "$HERE/report.sh" "$(dirname "$rec")" >/dev/null 2>&1 || true
  done
fi
echo "$record" > "${DEMO_TEARDOWN_OUT:-/dev/null}"

echo
if [ "$total" = 0 ]; then
  echo "down: nothing left (tag $TAG_KEY=$NAME, region $REGION). Teardown verified."
else
  echo "down: $total resource(s) remain; rerun, or delete by hand. See the counts above." >&2
  exit 1
fi
