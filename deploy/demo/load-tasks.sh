#!/usr/bin/env bash
# shellcheck disable=SC2016  # the container scripts are single-quoted on purpose
# The load proof's task definitions (SANDBOX=load). Sourced by up.sh in place of the gateway's and
# the agent's; it needs up.sh's helpers (register, LOGS, EXEC, EFSVOL, param_arn) and defines
# TD_AGENT and TD_LOADHOST.
#
# The agent task is the demo's size (0.5 vCPU, 1 GB) and runs the same image and flags, except:
#
#   --gateway-url   the scripted model in the load-host task, so a turn costs nothing
#   --bundle-cache-mb  small (LOAD_CACHE_MB, default 16), so eviction happens during the run
#   TMPDIR          a task volume shared with the sampler, so the sampler can size the bundle cache
#   --metrics-listen   loopback, for the sampler
#   --shard         /mnt/efs (the session store on EFS), or with LOAD_STORE=local a directory on the task's
#                   own ephemeral volume, to tell a cost of the network filesystem from one of the agent
#
# and a second container, the sampler, which shares the task's process namespace and network and writes
# one JSON line every two seconds to the log: what the kernel says about the agent process (resident
# and anonymous memory, threads, open fds), the bundle cache on disk, and the sockets. That is the only
# way to see the replica from inside its task, and it costs the replica a few hundred KB.

LOAD_CACHE_MB=${LOAD_CACHE_MB:-16}
LOAD_STORE=${LOAD_STORE:-efs}
case "$LOAD_STORE" in
  efs) STORE_DIR=/mnt/efs ;;
  local) STORE_DIR=/scratch/store ;;
  *) echo "LOAD_STORE must be efs or local, not $LOAD_STORE" >&2; exit 2 ;;
esac

AGENT_SCRIPT=$(cat <<'EOS'
until [ -w /scratch ]; do sleep 1; done
mkdir -p /scratch/tmp "$STORE_DIR"; export TMPDIR=/scratch/tmp; ulimit -c 0 2>/dev/null || true
/usr/local/bin/beyond-ai-agent serve --service --listen 0.0.0.0:8080 --gateway-url "$GATEWAY_URL" --model "$MODEL" \
  --grant-key "$GRANT_KEY_FLAG" --shard demo="$STORE_DIR" --bundle-cache-mb "$CACHE_MB" --metrics-listen 127.0.0.1:9100 &
wait $!
EOS
)

# sampler.sh sits beside this file so it can be run (and tested) outside a task.
SAMPLER_SCRIPT=$(cat "$HERE/sampler.sh")

TD_AGENT=$(jq -n --arg family "$P-agent" --arg exec "$(EXEC agent)" --argjson logs "$(LOGS agent)" --argjson slogs "$(LOGS sampler)" \
  --arg img "$ECR/$P-agent:latest" --arg simg "$ECR/$P-standin:latest" --argjson vol "$EFSVOL" \
  --arg script "$AGENT_SCRIPT" --arg sscript "$SAMPLER_SCRIPT" --arg gw "http://loadhost.$DNS:$LOADMODEL_PORT" \
  --arg model "$MODEL" --arg grant "$GRANT_KEY_FLAG" --arg seal "$(param_arn fleet-secret)" \
  --arg cache "$LOAD_CACHE_MB" --arg store_dir "$STORE_DIR" --arg lport "$LOADHOST_PORT" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"512", memory:"1024",
  pidMode:"task", executionRoleArn:$exec, volumes:($vol + [{name:"scratch"}]),
  containerDefinitions:[
    {name:"agent", image:$img, essential:true, stopTimeout:2, entryPoint:["/bin/sh","-c"], command:[$script],
     portMappings:[{containerPort:8080}],
     mountPoints:[{sourceVolume:"efs",containerPath:"/mnt/efs",readOnly:false},{sourceVolume:"scratch",containerPath:"/scratch",readOnly:false}],
     environment:[{name:"HOME",value:"/tmp"},{name:"GATEWAY_URL",value:$gw},{name:"MODEL",value:$model},
       {name:"GRANT_KEY_FLAG",value:$grant},{name:"CACHE_MB",value:$cache},{name:"STORE_DIR",value:$store_dir}],
     secrets:[{name:"AI_AGENT_SEAL_KEY_B64",valueFrom:$seal}],
     logConfiguration:$logs},
    {name:"sampler", image:$simg, essential:false, entryPoint:["/bin/sh","-c"], command:[$sscript],
     linuxParameters:{capabilities:{add:["SYS_PTRACE"]}},
     mountPoints:[{sourceVolume:"scratch",containerPath:"/scratch",readOnly:false}],
     environment:[{name:"LOADHOST_PORT",value:$lport}],
     logConfiguration:$slogs}]}' | register "$P-agent")

TD_LOADHOST=$(jq -n --arg family "$P-loadhost" --arg exec "$(EXEC loadhost)" --argjson logs "$(LOGS loadhost)" \
  --arg img "$ECR/$P-standin:latest" --arg tok "$(param_arn load-token)" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"512", memory:"1024",
  executionRoleArn:$exec,
  containerDefinitions:[
    {name:"loadhost", image:$img, essential:true, entryPoint:["/usr/local/bin/beyond-ai-demo"], command:["load-host"],
     portMappings:[{containerPort:8086},{containerPort:8083}],
     secrets:[{name:"DEMO_LOAD_TOKEN",valueFrom:$tok}],
     logConfiguration:$logs}]}' | register "$P-loadhost")
