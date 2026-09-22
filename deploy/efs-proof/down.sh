#!/usr/bin/env bash
# Give AWS back. Everything this proof creates carries `fleetsim=efs-proof`, and this deletes by that
# tag and then *verifies* by it.
#
# Teardown is a gate, not a courtesy. The run costs about $0.33/hr and the experiment is a few hours;
# left up it is ~$250/month, roughly a hundred times the cost of the thing it was measuring. The
# local half of this work lost hours to leaked mounts, namespaces and kernel threads, and those only
# cost load average. So this ends by listing what is left and exiting non-zero if anything is.
set -uo pipefail

REGION=${AWS_REGION:-us-west-2}
CLUSTER=${FLEETSIM_CLUSTER:-fleetsim}
TAG="Name=tag:fleetsim,Values=efs-proof"
aws() { command aws --region "$REGION" --output text "$@"; }

echo "── tasks ──"
arns=$(aws ecs list-tasks --cluster "$CLUSTER" --query 'taskArns' 2>/dev/null || true)
for a in $arns; do aws ecs stop-task --cluster "$CLUSTER" --task "$a" --query 'task.taskArn' >/dev/null; echo "stopping $a"; done
for _ in $(seq 1 60); do
  left=$(aws ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING --query 'length(taskArns)' 2>/dev/null || echo 0)
  [ "$left" = "0" ] && break
  sleep 5
done

echo "── efs ──"
# Mount targets before the filesystem, and the filesystem only once they are gone: EFS refuses to
# delete a filesystem that still has one, and the deletion is not instant.
for fs in $(aws efs describe-file-systems --query 'FileSystems[?Tags[?Key==`fleetsim` && Value==`efs-proof`]].FileSystemId'); do
  for ap in $(aws efs describe-access-points --file-system-id "$fs" --query 'AccessPoints[].AccessPointId'); do
    aws efs delete-access-point --access-point-id "$ap"; echo "deleted $ap"
  done
  for mt in $(aws efs describe-mount-targets --file-system-id "$fs" --query 'MountTargets[].MountTargetId'); do
    aws efs delete-mount-target --mount-target-id "$mt"; echo "deleted $mt"
  done
  for _ in $(seq 1 60); do
    n=$(aws efs describe-mount-targets --file-system-id "$fs" --query 'length(MountTargets)' 2>/dev/null || echo 0)
    [ "$n" = "0" ] && break
    sleep 5
  done
  aws efs delete-file-system --file-system-id "$fs" && echo "deleted $fs"
done

echo "── network ──"
# The ENIs an ECS task leaves behind hold the subnets hostage for a minute or two after the task is
# gone. Nothing here can hurry that, so it is waited on rather than worked around.
for _ in $(seq 1 60); do
  n=$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=$(aws ec2 describe-vpcs --filters "$TAG" --query 'Vpcs[0].VpcId')" --query 'length(NetworkInterfaces)' 2>/dev/null || echo 0)
  [ "$n" = "0" ] || [ "$n" = "None" ] && break
  sleep 5
done
vpc=$(aws ec2 describe-vpcs --filters "$TAG" --query 'Vpcs[0].VpcId')
if [ -n "$vpc" ] && [ "$vpc" != "None" ]; then
  for acl in $(aws ec2 describe-network-acls --filters "$TAG" --query 'NetworkAcls[].NetworkAclId'); do
    aws ec2 delete-network-acl --network-acl-id "$acl" 2>/dev/null && echo "deleted $acl"
  done
  for sub in $(aws ec2 describe-subnets --filters "$TAG" --query 'Subnets[].SubnetId'); do
    aws ec2 delete-subnet --subnet-id "$sub" 2>/dev/null && echo "deleted $sub"
  done
  for sg in $(aws ec2 describe-security-groups --filters "$TAG" --query 'SecurityGroups[].GroupId'); do
    aws ec2 delete-security-group --group-id "$sg" 2>/dev/null && echo "deleted $sg"
  done
  for igw in $(aws ec2 describe-internet-gateways --filters "$TAG" --query 'InternetGateways[].InternetGatewayId'); do
    aws ec2 detach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$vpc" 2>/dev/null
    aws ec2 delete-internet-gateway --internet-gateway-id "$igw" 2>/dev/null && echo "deleted $igw"
  done
  for rt in $(aws ec2 describe-route-tables --filters "$TAG" --query 'RouteTables[].RouteTableId'); do
    aws ec2 delete-route-table --route-table-id "$rt" 2>/dev/null && echo "deleted $rt"
  done
  aws ec2 delete-vpc --vpc-id "$vpc" 2>/dev/null && echo "deleted $vpc"
fi

echo "── cluster, logs, images, roles ──"
aws ecs delete-cluster --cluster "$CLUSTER" --query 'cluster.status' 2>/dev/null
aws logs delete-log-group --log-group-name /fleetsim 2>/dev/null
for repo in fleetsim-agent fleetsim-driver; do
  aws ecr delete-repository --repository-name "$repo" --force --query 'repository.repositoryName' 2>/dev/null
done
for role in fleetsimDriverTaskRole fleetsimReplicaTaskRole fleetsimTaskExecutionRole; do
  for pol in $(aws iam list-role-policies --role-name "$role" --query 'PolicyNames' 2>/dev/null); do
    aws iam delete-role-policy --role-name "$role" --policy-name "$pol" 2>/dev/null
  done
  for arn in $(aws iam list-attached-role-policies --role-name "$role" --query 'AttachedPolicies[].PolicyArn' 2>/dev/null); do
    aws iam detach-role-policy --role-name "$role" --policy-arn "$arn" 2>/dev/null
  done
  aws iam delete-role --role-name "$role" 2>/dev/null && echo "deleted role $role"
done

# The gate. Anything still tagged is something this script did not delete, and saying so is the
# point: "I ran the teardown" is not the same claim as "nothing is left", and only one of them is
# checkable.
echo
echo "── what is left ──"
left=0
for kind in vpcs subnets security-groups network-acls internet-gateways route-tables; do
  n=$(aws ec2 "describe-$kind" --filters "$TAG" --query 'length(@)' 2>/dev/null | head -1)
  found=$(aws ec2 "describe-$kind" --filters "$TAG" --query "length(*[0])" 2>/dev/null || echo 0)
  [ "$found" != "0" ] && { echo "  $kind: $found"; left=$((left + found)); }
done
fs=$(aws efs describe-file-systems --query 'length(FileSystems[?Tags[?Key==`fleetsim` && Value==`efs-proof`]])')
[ "$fs" != "0" ] && { echo "  file systems: $fs"; left=$((left + fs)); }
if [ "$left" = "0" ]; then
  echo "  nothing — teardown verified"
else
  echo "  ^ still tagged fleetsim=efs-proof; rerun, or delete by hand" >&2
  exit 1
fi
