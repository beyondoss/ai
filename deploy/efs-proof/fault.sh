#!/usr/bin/env bash
# Perform a fault on a replica the simulator does not own.
#
# Invoked by `fleet-sim --fault-cmd` as `fault.sh <action> <replica>`, where <replica> is `r1`, `r2`,
# … in the order the `--replica` flags were given. Every action is synchronous: it returns only once
# the fault has actually happened, because the simulator's next assertion is only meaningful then.
#
# The mapping from a replica *name* to AWS resources is by tag, never by a baked-in id — a task ARN
# changes on every restart, so anything cached here would be wrong by the second scenario:
#
#   task    ecs:ListTasks + DescribeTasks --include TAGS, matching tag replica=<name>
#   subnet  the subnet that task's ENI is in
#   NACL    ec2:DescribeNetworkAcls, matching tag replica=<name>
#
# ## kill vs term
#
# `term` is plain StopTask: SIGTERM, then the task definition's stopTimeout before SIGKILL. That is
# exactly the deploy case the agent is supposed to survive, and it is what ECS does on a rolling
# deploy, so it is the real thing rather than a stand-in.
#
# `kill` is the case the epoch fence exists for — a machine that dies with no chance to seal — and
# Fargate has no "SIGKILL this task now" API. StopTask always sends SIGTERM first, which would let
# the replica seal its segment and turn C1/C2 into a test of the graceful path wearing the name of
# the hard one. So `kill` goes in through ECS Exec and SIGKILLs the agent directly. That needs the
# agent *not* to be PID 1 (the kernel refuses SIGKILL to a PID namespace's init from inside it),
# which is why the replica task definition overrides the image's entrypoint with a shell that
# backgrounds the agent and forwards SIGTERM to it. The shipped image still execs — the override
# lives in the task definition, not in `Dockerfile.agent`.
#
# ## partition
#
# A NACL deny on the replica's own subnet, for the EFS mount target address only. NACLs are
# stateless, so an already-established NFS connection is cut on the next packet; a security group
# would leave the tracked flow up and produce a "the design strands sessions" result that was really
# a rig artifact. The mount target deliberately lives in the *driver's* subnet, so replica → EFS
# traffic crosses a subnet boundary and is filtered at all, while driver → EFS never is.
set -euo pipefail

CLUSTER=${FLEETSIM_CLUSTER:-fleetsim}
REGION=${AWS_REGION:-us-west-2}
EFS_IP=${FLEETSIM_EFS_IP:?FLEETSIM_EFS_IP must be the EFS mount target address}
RULE=${FLEETSIM_DENY_RULE:-50}

action=${1:?usage: fault.sh <kill|term|stop|partition|heal|restart> <replica>}
replica=${2:?usage: fault.sh <kill|term|stop|partition|heal|restart> <replica>}

aws() { command aws --region "$REGION" --output text "$@"; }

# The running task tagged replica=<name>, or empty.
task_arn() {
  local arns
  arns=$(aws ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING --query 'taskArns' || true)
  [ -n "$arns" ] || return 0
  # shellcheck disable=SC2086
  aws ecs describe-tasks --cluster "$CLUSTER" --tasks $arns --include TAGS \
    --query "tasks[?tags[?key=='replica' && value=='$replica']].taskArn"
}

acl_id() {
  aws ec2 describe-network-acls \
    --filters "Name=tag:fleetsim,Values=efs-proof" "Name=tag:replica,Values=$replica" \
    --query 'NetworkAcls[0].NetworkAclId'
}

acl_subnet() {
  aws ec2 describe-network-acls \
    --filters "Name=tag:fleetsim,Values=efs-proof" "Name=tag:replica,Values=$replica" \
    --query 'NetworkAcls[0].Associations[0].SubnetId'
}

# Wait until the task tagged <replica> is gone from the RUNNING set.
wait_stopped() {
  local arn=$1
  for _ in $(seq 1 120); do
    local st
    st=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --query 'tasks[0].lastStatus' || echo STOPPED)
    [ "$st" = "STOPPED" ] && return 0
    sleep 1
  done
  echo "fault: $replica did not reach STOPPED" >&2
  return 1
}

case "$action" in
  term)
    # StopTask and return. ECS sends SIGTERM to the task's init — the wrapper shell in the replica's
    # task definition, which forwards it to the agent — and only SIGKILLs after the container's
    # `stopTimeout`. The drain window is everything between those two, so this must *not* wait: the
    # first EFS run did, and C7 graded a replica that had already exited (`/readyz answered 0`).
    arn=$(task_arn)
    [ -n "$arn" ] || { echo "fault: no running task tagged replica=$replica" >&2; exit 1; }
    aws ecs stop-task --cluster "$CLUSTER" --task "$arn" --reason "fleet-sim term" --query 'task.taskArn' >/dev/null
    ;;

  stop)
    # The same signal, and this one is over when the replica is gone.
    arn=$(task_arn)
    [ -n "$arn" ] || exit 0
    aws ecs stop-task --cluster "$CLUSTER" --task "$arn" --reason "fleet-sim stop" --query 'task.taskArn' >/dev/null
    wait_stopped "$arn"
    ;;

  kill)
    arn=$(task_arn)
    [ -n "$arn" ] || { echo "fault: no running task tagged replica=$replica" >&2; exit 1; }
    # SIGKILL the agent itself. The wrapper shell (PID 1) then falls out of its wait loop and
    # exits, the essential container is gone, and ECS stops the task.
    #
    # Matched against `/proc/<pid>/comm` exactly, rather than with `pgrep`. `pgrep -f
    # beyond-ai-agent` matches the wrapper shell too — the agent's command line is inside it — and
    # on this image that is pids 1, 8 and 9, the first of which the kernel refuses to SIGKILL from
    # inside its own namespace. BusyBox's `pgrep -x` matches nothing here at all. Either way the
    # fault quietly does nothing and the scenario reports that a hard kill failed to fence anybody,
    # which is a claim about the design made out of a bad pattern. `comm` is exact and is in every
    # kernel, so it needs no tool to agree with.
    # Issued more than once, and checked in between. The exec is bounded because an `--interactive`
    # session whose stdin is /dev/null does not always notice it is over — the signal lands in
    # milliseconds and the session then sat for twenty minutes — and it is *retried* because the
    # ExecuteCommandAgent in a freshly started task comes up a while after the agent does. A replica
    # the simulator restored seconds ago answers `/readyz` long before it can be exec'd into, so a
    # single attempt is a coin flip, and the way it loses is silence: a kill that did nothing, then
    # a scenario reporting that a hard kill failed to fence anybody.
    for attempt in 1 2 3 4; do
      timeout 45 aws ecs execute-command --cluster "$CLUSTER" --task "$arn" --container agent --interactive \
        --command "/bin/sh -c 'for p in /proc/[0-9]*; do [ \"\$(cat \$p/comm 2>/dev/null)\" = beyond-ai-agent ] && kill -9 \${p#/proc/}; done; true'" >/dev/null 2>&1 || true
      for _ in $(seq 1 15); do
        st=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --query 'tasks[0].lastStatus' || echo STOPPED)
        [ "$st" = "STOPPED" ] && exit 0
        sleep 2
      done
      echo "fault: kill attempt $attempt did not land on $replica, retrying" >&2
    done
    echo "fault: $replica did not reach STOPPED" >&2
    exit 1
    ;;

  partition)
    acl=$(acl_id)
    [ -n "$acl" ] && [ "$acl" != "None" ] || { echo "fault: no NACL tagged replica=$replica" >&2; exit 1; }
    # Both directions: stateless means the return path is filtered independently, and denying only
    # egress would leave EFS's retransmits arriving at a client that cannot answer — a half-cut that
    # behaves like neither a partition nor a healthy mount.
    for dir in --egress --ingress; do
      aws ec2 create-network-acl-entry --network-acl-id "$acl" --rule-number "$RULE" \
        --protocol -1 --rule-action deny $dir --cidr-block "$EFS_IP/32" 2>/dev/null \
        || aws ec2 replace-network-acl-entry --network-acl-id "$acl" --rule-number "$RULE" \
             --protocol -1 --rule-action deny $dir --cidr-block "$EFS_IP/32"
    done
    ;;

  heal)
    acl=$(acl_id)
    [ -n "$acl" ] && [ "$acl" != "None" ] || { echo "fault: no NACL tagged replica=$replica" >&2; exit 1; }
    for dir in --egress --ingress; do
      aws ec2 delete-network-acl-entry --network-acl-id "$acl" --rule-number "$RULE" $dir 2>/dev/null || true
    done
    ;;

  address)
    # Where this replica answers now. Asked for rather than assumed, because a replaced Fargate task
    # draws a fresh private address from its subnet and nothing in ECS will pin one.
    arn=$(task_arn)
    [ -n "$arn" ] || { echo "fault: no running task tagged replica=$replica" >&2; exit 1; }
    ip=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" \
      --query 'tasks[0].attachments[0].details[?name==`privateIPv4Address`].value|[0]')
    [ -n "$ip" ] && [ "$ip" != "None" ] || { echo "fault: $replica has no address yet" >&2; exit 1; }
    echo "$ip:${FLEETSIM_PORT:-8080}"
    ;;

  restart)
    # A deploy replacing a task, not a process coming back: a brand new task from the same
    # definition (all replicas share one — they differ by tag and subnet, not by config), in the
    # same subnet, tagged the same way. It comes back on a *different* private address, which is why
    # the simulator asks `address` afterwards rather than assuming the one it had.
    subnet=$(acl_subnet)
    sg=${FLEETSIM_SG:?FLEETSIM_SG must be the task security group}
    # Idempotent, and not as a nicety: the simulator restores the fleet before every scenario, so
    # `restart` is asked of replicas that are dead, alive, and — after the drain scenario — still
    # finishing what they owned. Two tasks tagged the same would make `address` ambiguous and put a
    # second writer on the shard, so whatever is there goes first.
    old=$(task_arn)
    if [ -n "$old" ]; then
      aws ecs stop-task --cluster "$CLUSTER" --task "$old" --reason "fleet-sim restart" >/dev/null
      wait_stopped "$old"
    fi
    arn=$(aws ecs run-task --cluster "$CLUSTER" --launch-type FARGATE \
      --task-definition "${FLEETSIM_TASKDEF:-fleetsim-replica}" --enable-execute-command \
      --network-configuration "awsvpcConfiguration={subnets=[$subnet],securityGroups=[$sg],assignPublicIp=ENABLED}" \
      --tags "key=replica,value=$replica" "key=fleetsim,value=efs-proof" \
      --query 'tasks[0].taskArn')
    [ -n "$arn" ] || { echo "fault: run-task for $replica returned nothing" >&2; exit 1; }
    for _ in $(seq 1 180); do
      st=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" --query 'tasks[0].lastStatus')
      [ "$st" = "RUNNING" ] && break
      [ "$st" = "STOPPED" ] && { echo "fault: $replica stopped while starting" >&2; exit 1; }
      sleep 1
    done
    [ "$st" = "RUNNING" ] || { echo "fault: $replica did not reach RUNNING" >&2; exit 1; }
    # And wait for the way back *in*. A task is RUNNING, and answers its health checks, before its
    # ExecuteCommandAgent is up — and that agent is how `kill` reaches the process. A replica that
    # is not yet killable is not yet restarted, however healthy it looks from outside.
    for _ in $(seq 1 90); do
      # Piped at each filter. Chaining two filter projections with `.` gives a list of lists that
      # `|[0][0]` does not reach into, and the query comes back `None` — which reads as "the agent
      # never started" rather than "the query was wrong", and fails the restart of a replica that
      # was in fact perfectly ready.
      ea=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$arn" \
        --query 'tasks[0].containers[?name==`agent`]|[0].managedAgents[?name==`ExecuteCommandAgent`]|[0].lastStatus')
      [ "$ea" = "RUNNING" ] && exit 0
      sleep 2
    done
    echo "fault: $replica came up but its exec agent did not ($ea)" >&2
    exit 1
    ;;

  capacity)
    # Not a fault: how many replicas this fleet has. Used by the teardown check.
    aws ecs list-tasks --cluster "$CLUSTER" --desired-status RUNNING --query 'length(taskArns)'
    ;;

  *)
    echo "fault: unknown action $action" >&2
    exit 2
    ;;
esac
