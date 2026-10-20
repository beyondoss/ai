#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2016  # a sourced library: its variables are the callers, and queries are single-quoted on purpose
# Shared by up.sh, down.sh and demo.sh. Source it; it runs nothing.
#
# Everything the demo creates carries the tag `beyond-ai-demo=<name>`, and every name starts with
# `beyond-ai-demo-<name>`, so "is it mine?" is a lookup and never a guess.

REGION=${AWS_REGION:-us-west-2}
NAME=${DEMO_NAME:-demo}
case "$NAME" in
  *[!a-z0-9-]* | "" | -*) echo "DEMO_NAME must be lowercase letters, digits and dashes" >&2; exit 2 ;;
esac
TAG_KEY=beyond-ai-demo
P=beyond-ai-demo-$NAME           # name prefix for IAM, ECS, ECR, logs, EFS, ...
SSM=/beyond-ai-demo/$NAME        # parameter prefix: the only place a key is ever stored
DNS=$NAME.local                  # the Cloud Map private namespace: gateway.<dns>, agent.<dns>, ...
CLUSTER=$P
LOG_GROUP=/beyond-ai-demo/$NAME
CIDR=10.88.0.0/16
SUBNET_CIDR=10.88.0.0/24
AZ=${DEMO_AZ:-${REGION}a}
AGENT_PORT=8080                  # plain ws: the demo client; the only port open to the caller
AGENT_TLS_PORT=8443              # https: where the sandbox calls back (job_ended)
GATEWAY_PORT=8080
SANDBOX_PORT=8081
# Where the sandbox runs: `fargate` (default, one shared sandbox service) or `e2b` (one E2B microVM per
# session, made by demo.sh). The same exec-stub either way.
SANDBOX=${SANDBOX:-fargate}
case "$SANDBOX" in
  fargate | e2b) ;;
  *) echo "SANDBOX must be fargate or e2b" >&2; exit 2 ;;
esac
TAGSPEC="{Key=$TAG_KEY,Value=$NAME}"

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)

# shellcheck source=e2b.sh
. "$HERE/e2b.sh"

aws() { command aws --region "$REGION" --output text "$@"; }
awsj() { command aws --region "$REGION" --output json "$@"; }
# aws, with the CLI's "None" for a missing value turned into the empty string.
q() {
  local v
  v=$(aws "$@") || return 1
  [ "$v" = None ] && v=""
  printf '%s' "$v"
}
say() { echo "── $* ──"; }

# The repo's own tools (sandbox, grant minter, chat client), built once from this checkout.
demo_bin() {
  (cd "$ROOT" && cargo build -q -p beyond-ai-demo >&2) || return 1
  local dir
  dir=$(cd "$ROOT" && cargo metadata --format-version 1 --no-deps 2>/dev/null | jq -r .target_directory)
  echo "$dir/debug/beyond-ai-demo"
}

ssm_get() { command aws --region "$REGION" --output text ssm get-parameter --with-decryption --name "$SSM/$1" --query Parameter.Value; }

account() { aws sts get-caller-identity --query Account; }

# The agent's current public address, or empty. Looked up every time: a replaced task is a new IP.
agent_ip() { # [task arn: a specific task, else the first running one]
  local task=${1:-} eni
  [ -n "$task" ] || task=$(q ecs list-tasks --cluster "$CLUSTER" --service-name "$P-agent" --desired-status RUNNING --query 'taskArns[0]') || return 0
  [ -n "$task" ] || return 0
  eni=$(q ecs describe-tasks --cluster "$CLUSTER" --tasks "$task" \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value|[0]') || return 0
  [ -n "$eni" ] || return 0
  q ec2 describe-network-interfaces --network-interface-ids "$eni" --query 'NetworkInterfaces[0].Association.PublicIp'
}
