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
#   EFS                        the agent's sessions, sealed per tenant; only the agent's task role may mount it
#                              (IAM + a file-system policy), through its access point, over TLS, and only the
#                              agent's security group reaches the mount target: the sandbox has no route to it
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
if [ "$USE_E2B" = 1 ]; then
  e2b_need_key || exit 2
  # The key works. (VMs are made per session by demo.sh, so there is nothing to create here.)
  e2b_count >/dev/null || { echo "up: E2B's API did not accept E2B_API_KEY" >&2; exit 2; }
  BROWSER_TEMPLATE=
fi
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
if [ "$LOAD" = 1 ]; then
  : # the load proof talks to a scripted model: no provider key, no gateway
elif [ -n "$KEY" ]; then
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

# The load proof's bundle host takes uploads and reports stats only with this token.
if [ "$LOAD" = 1 ]; then
  MODEL=claude-test
  param_exists load-token || openssl rand -hex 24 | tr -d "\n" | put_param load-token SecureString
fi

# The Fargate sandbox is on a private subnet but runs agent-controlled code and listens beyond loopback,
# so execd requires a bearer: a per-stack secret that only the sandbox task (as EXECD_BEARER) and the
# grant minter (demo.sh) can read. Created once, and kept by reruns.
if [ "$USE_FARGATE" = 1 ]; then
  param_exists sandbox-bearer || openssl rand -hex 24 | tr -d '\n' | put_param sandbox-bearer SecureString
fi

# For the use cases: the secret the stand-in managed-MCP endpoint shares with the grant minter, and the
# "third party's" real key, which only the stand-in's task can read. Created once, and kept by reruns.
if [ "$USECASES" = 1 ]; then
  param_exists managed-secret || openssl rand -base64 32 | tr -d '\n' | put_param managed-secret SecureString
  param_exists provider-secret || printf 'bsk_live_demo_%s' "$(openssl rand -hex 24)" | put_param provider-secret SecureString
  # The exec front (below) makes a session's E2B microVM on its first request, so it holds the key; only
  # its task can read it.
  printf '%s' "$E2B_API_KEY" | put_param e2b-key SecureString
fi

# For `demo.sh oidc`: the bmc_v1 key (as above), the key tenants' keys derive from (refresh tokens are sealed
# under them), the control plane's credential, the broker's client secret at the identity provider, two
# users' passwords, and the provider's realm (built from oidc/realm.template.json with those secrets).
# Created once, kept by reruns; only the stack's own task roles can read them.
if [ "$OIDC" = 1 ]; then
  param_exists managed-secret || openssl rand -base64 32 | tr -d '\n' | put_param managed-secret SecureString
  param_exists broker-master || openssl rand -base64 32 | tr -d '\n' | put_param broker-master SecureString
  param_exists broker-admin || printf '%s' "$(openssl rand -hex 24)" | put_param broker-admin SecureString
  param_exists oidc-client-secret || printf '%s' "$(openssl rand -hex 24)" | put_param oidc-client-secret SecureString
  param_exists oidc-users || jq -nc --arg a "$(openssl rand -hex 12)" --arg b "$(openssl rand -hex 12)" '{alice:$a,bob:$b}' | put_param oidc-users SecureString
  OIDC_USERS=$(ssm_get oidc-users)
  sed -e "s|__CLIENT_SECRET__|$(ssm_get oidc-client-secret)|" \
    -e "s|__REDIRECT_URI__|http://oidc.$DNS:$OIDC_CONTROL_PORT/oauth/callback|" \
    -e "s|__ALICE_PASSWORD__|$(jq -r .alice <<<"$OIDC_USERS")|" -e "s|__BOB_PASSWORD__|$(jq -r .bob <<<"$OIDC_USERS")|" \
    "$HERE/oidc/realm.template.json" | jq -c . | put_param oidc-realm SecureString
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
  printf 'subjectAltName=DNS:agent.%s,DNS:*.sslip.io\nbasicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n' "$DNS" > "$WORK/ext.cnf"
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
ensure_image() { # repo, dockerfile, [build target]
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
  docker build --platform linux/amd64 -q -f "$ROOT/$file" ${3:+--target "$3"} -t "$ECR/$repo:latest" "$ROOT" >/dev/null
  docker push -q "$ECR/$repo:latest" >/dev/null
  made "image $repo:latest"
  IMAGES_CHANGED=1
}
IMAGES_CHANGED=
[ "$LOAD" = 1 ] || ensure_image "$P-gateway" Dockerfile
ensure_image "$P-agent" Dockerfile.agent
[ "$USE_FARGATE" = 0 ] || ensure_image "$P-sandbox" deploy/demo/Dockerfile.sandbox
{ [ "$USECASES" = 1 ] || [ "$OIDC" = 1 ] || [ "$LOAD" = 1 ]; } && ensure_image "$P-standin" deploy/demo/Dockerfile.standin
[ "$USECASES" = 0 ] || ensure_image "$P-front" deploy/demo/Dockerfile.sandbox front

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
SG_AGENT=$(sg_for "$P-agent" "beyond-ai-demo agent replicas")
SG_SANDBOX=$(sg_for "$P-sandbox" "beyond-ai-demo sandbox")
SG_EFS=$(sg_for "$P-efs" "beyond-ai-demo efs mount target")
# Three groups. The agent's holds the only identity the session store admits; the sandbox runs
# agent-controlled code and has no route to the store; every other task (gateway, stand-ins, ...) sits
# in the third. The only way in from outside is the agent's websocket port, from the caller's address.
sg_rule() { # group, extra authorize args...; counts only a rule that was new
  local out
  if out=$(command aws --region "$REGION" ec2 authorize-security-group-ingress --group-id "$1" "${@:2}" 2>&1); then
    made "ingress rule on $1"
  elif [[ $out != *InvalidPermission.Duplicate* ]]; then
    echo "$out" >&2
    return 1
  fi
}
sg_rule "$SG_TASKS" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_TASKS},{GroupId=$SG_AGENT}]"
sg_rule "$SG_AGENT" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_TASKS}]"
# The agent calls the sandbox's exec endpoint (and the exec front); the sandbox calls back (job_ended)
# over TLS and may reach nothing else of ours.
sg_rule "$SG_SANDBOX" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_AGENT}]"
sg_rule "$SG_AGENT" --protocol tcp --port "$AGENT_TLS_PORT" --source-group "$SG_SANDBOX"
# The session store admits the agent's group alone.
sg_rule "$SG_EFS" --protocol tcp --port 2049 --source-group "$SG_AGENT"
CALLER=${DEMO_ALLOW_CIDR:-}
if [ -z "$CALLER" ]; then
  CALLER="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32"
fi
sg_rule "$SG_AGENT" --protocol tcp --port "$AGENT_PORT" --cidr "$CALLER"
# E2B's microVMs call back (job_ended) from addresses nobody can list, so the https port is open to
# the world in that mode: it accepts only a callback token the agent's fleet key minted.
[ "$USE_E2B" = 0 ] || sg_rule "$SG_AGENT" --protocol tcp --port "$AGENT_TLS_PORT" --cidr 0.0.0.0/0
# The stand-in managed-MCP endpoint is a public service, like Beyond's would be: it answers only a
# request carrying a valid bmc_v1 token. (Public, so the agent's MCP egress policy needs no exception.)
{ [ "$USECASES" = 1 ] || [ "$OIDC" = 1 ]; } && sg_rule "$SG_TASKS" --protocol tcp --port "$STANDIN_PORT" --cidr 0.0.0.0/0
# The broker's control plane and the identity provider's login page: the caller's address alone (the
# scripted user signs in from here; in a real deployment the browser would, and the callback is public).
if [ "$OIDC" = 1 ]; then
  sg_rule "$SG_TASKS" --protocol tcp --port "$OIDC_CONTROL_PORT" --cidr "$CALLER"
  sg_rule "$SG_TASKS" --protocol tcp --port "$OIDC_IDP_PORT" --cidr "$CALLER"
fi
# The load proof's bundle host: the replica fetches bundles from it at its public address, as it would
# from S3. Uploads and stats need the token; a bundle is only served by the hash it is named by.
[ "$LOAD" = 0 ] || sg_rule "$SG_TASKS" --protocol tcp --port "$LOADHOST_PORT" --cidr 0.0.0.0/0

# ---------------------------------------------------------------------------------------------
# The use cases' agent folders are deployed as bundles: one tar per folder, named by its hash, in a
# private bucket that demo.sh uploads to and hands the replica presigned links to. down.sh empties and
# deletes it.
if [ "$USECASES" = 1 ] || [ "$OIDC" = 1 ]; then
  say "bundles bucket"
  BUCKET=$(bundle_bucket)
  if ! command aws --region "$REGION" s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
    made "bucket $BUCKET"
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" \
    --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging "TagSet=[{Key=$TAG_KEY,Value=$NAME}]"
  # The browser agent's VMs boot from a template with the browser stack and the rustwright-mcp binary in it
  # (found by name if an earlier run built it; see e2b_browser_template). It belongs to the account, so
  # `down.sh` keeps it.
  [ "$USE_E2B" != 1 ] || BROWSER_TEMPLATE=$(e2b_browser_template) || exit 2
fi

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
# A mount target keeps the groups it was made with; say again which one it has.
aws efs modify-mount-target-security-groups --mount-target-id "$MT" --security-groups "$SG_EFS"
# The agent runs as uid 10001 (Dockerfile.agent); the access point pins every caller to it.
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
# reads its own seal and TLS keys; the sandbox's reads its bearer (and the front's E2B key). No role can
# read the whole prefix.
[ "$LOAD" = 1 ] || role gateway-exec "$PROVIDER-key id-signing-key"
role agent-exec "fleet-secret tls-key"
[ "$USE_FARGATE" = 0 ] || role sandbox-exec "sandbox-bearer$([ "$USECASES" = 0 ] || echo ' e2b-key')"
# The stand-in's role is the only one that can read the "third party's" key.
[ "$USECASES" = 0 ] || role standin-exec "managed-secret provider-secret"
[ "$LOAD" = 0 ] || role loadhost-exec "load-token"
# The identity provider's realm (its users' passwords) and the broker's keys: this stack's task alone.
[ "$OIDC" = 0 ] || role oidc-exec "managed-secret broker-master broker-admin oidc-client-secret oidc-realm"
# The agent's task role. It is the one identity the session store admits (the file-system policy below
# names it), and with ECS Exec on (`demo.sh assistant` and `demo.sh oidc` scan the agent process's memory
# through it) it may also open the SSM message channel and nothing else.
if ! aws iam get-role --role-name "$P-agent-task" >/dev/null 2>&1; then
  aws iam create-role --role-name "$P-agent-task" --assume-role-policy-document "$TRUST" --tags "Key=$TAG_KEY,Value=$NAME" >/dev/null
  made "role $P-agent-task"
  sleep 10
fi
AGENT_TASK_ROLE="arn:aws:iam::$ACCOUNT:role/$P-agent-task"
FS_ARN="arn:aws:elasticfilesystem:$REGION:$ACCOUNT:file-system/$FS"
AP_ARN="arn:aws:elasticfilesystem:$REGION:$ACCOUNT:access-point/$AP"
aws iam put-role-policy --role-name "$P-agent-task" --policy-name mount-sessions --policy-document \
  "$(jq -n --arg fs "$FS_ARN" --arg ap "$AP_ARN" '{Version:"2012-10-17",Statement:[{Effect:"Allow",Action:["elasticfilesystem:ClientMount","elasticfilesystem:ClientWrite"],Resource:$fs,Condition:{StringEquals:{"elasticfilesystem:AccessPointArn":$ap}}}]}')"
if [ "$EXECSCAN" = 1 ]; then
  aws iam put-role-policy --role-name "$P-agent-task" --policy-name ecs-exec --policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["ssmmessages:CreateControlChannel","ssmmessages:CreateDataChannel","ssmmessages:OpenControlChannel","ssmmessages:OpenDataChannel"],"Resource":"*"}]}'
fi
# The session store's own policy: only that role mounts it, only through the access point (so never the
# file system root), and never without TLS. Anything else, a sandbox that found the mount target
# included, is refused by EFS itself. (Applied every run: the document is the same, so put is a no-op.)
aws efs put-file-system-policy --file-system-id "$FS" --policy "$(jq -n --arg role "$AGENT_TASK_ROLE" --arg fs "$FS_ARN" --arg ap "$AP_ARN" '{
  Version:"2012-10-17",
  Statement:[
    {Sid:"AgentMountsThroughItsAccessPoint", Effect:"Allow", Principal:{AWS:$role},
     Action:["elasticfilesystem:ClientMount","elasticfilesystem:ClientWrite"], Resource:$fs,
     Condition:{StringEquals:{"elasticfilesystem:AccessPointArn":$ap}}},
    {Sid:"NeverWithoutTls", Effect:"Deny", Principal:{AWS:"*"}, Action:"*", Resource:$fs,
     Condition:{Bool:{"aws:SecureTransport":"false"}}}]}')" >/dev/null
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
  '[{name:"efs",efsVolumeConfiguration:{fileSystemId:$fs,transitEncryption:"ENABLED",authorizationConfig:{accessPointId:$ap,iam:"ENABLED"}}}]')

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

if [ "$LOAD" = 0 ]; then
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
fi

# The agent is NOT pid 1: pid 1 is a shell that never forwards SIGTERM. A stopped task is therefore
# SIGKILLed two seconds later, agent included, with no chance to seal its segment or release its
# lock: the hard kill `demo.sh kill` is about, done with nothing but `aws ecs stop-task`.
AGENT_SCRIPT='printf "%s" "$TLS_CERT" > /tmp/leaf.pem; printf "%s" "$TLS_KEY" > /tmp/leaf.key; ulimit -c 0 2>/dev/null || true
/usr/local/bin/beyond-ai-agent serve --service --listen 0.0.0.0:8080 --tls-listen 0.0.0.0:8443 \
  --tls-cert /tmp/leaf.pem --tls-key /tmp/leaf.key --gateway-url "$GATEWAY_URL" --model "$MODEL" \
  --notify-origin "$NOTIFY_ORIGIN" --grant-key "$GRANT_KEY_FLAG" --shard demo=/mnt/efs &
wait $!'
# With E2B the sandbox is on the public internet: the callback origin is this task's own public
# address, as a name the demo's CA has a certificate for (<a-b-c-d>.sslip.io resolves to a.b.c.d).
# Computed at every start, so a replacement task announces its new address.
if [ "$USE_E2B" = 1 ]; then
  AGENT_SCRIPT='until IP=$(wget -q -T 5 -O - http://checkip.amazonaws.com 2>/dev/null | tr -d "[:space:]") && [ -n "$IP" ]; do sleep 1; done
NOTIFY_ORIGIN="https://$(printf %s "$IP" | tr . -).sslip.io:8443"; echo "notify origin: $NOTIFY_ORIGIN"
'"$AGENT_SCRIPT"
fi
if [ "$LOAD" = 1 ]; then
  # shellcheck source=load-tasks.sh
  . "$HERE/load-tasks.sh"
else
TD_AGENT=$(jq -n --arg family "$P-agent" --arg exec "$(EXEC agent)" --arg taskrole "$AGENT_TASK_ROLE" --argjson logs "$(LOGS agent)" \
  --arg img "$ECR/$P-agent:latest" --argjson vol "$EFSVOL" --arg script "$AGENT_SCRIPT" \
  --arg gw "http://gateway.$DNS:$GATEWAY_PORT" --arg model "$MODEL" --arg origin "https://agent.$DNS:$AGENT_TLS_PORT" \
  --arg execscan "$EXECSCAN" --arg grant "$GRANT_KEY_FLAG" --arg cert "$TLS_CERT" --arg seal "$(param_arn fleet-secret)" --arg tlskey "$(param_arn tls-key)" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"512", memory:"1024",
  executionRoleArn:$exec, volumes:$vol,
  containerDefinitions:[
    {name:"agent", image:$img, essential:true, stopTimeout:2, entryPoint:["/bin/sh","-c"], command:[$script],
     portMappings:[{containerPort:8080},{containerPort:8443}],
     mountPoints:[{sourceVolume:"efs",containerPath:"/mnt/efs",readOnly:false}],
     environment:[{name:"HOME",value:"/tmp"},{name:"GATEWAY_URL",value:$gw},{name:"MODEL",value:$model},
       {name:"NOTIFY_ORIGIN",value:$origin},{name:"GRANT_KEY_FLAG",value:$grant},{name:"TLS_CERT",value:$cert}],
     secrets:[{name:"AI_AGENT_SEAL_KEY_B64",valueFrom:$seal},{name:"TLS_KEY",valueFrom:$tlskey}],
     logConfiguration:$logs}]} + {taskRoleArn:$taskrole}
  # The use cases scan the agent process'"'"'s memory through ECS Exec, which Linux allows a non-parent
  # only with CAP_SYS_PTRACE (Yama scope 1 on Fargate): granted for that, in this mode only.
  | if $execscan == "1" then .containerDefinitions[0].linuxParameters = {capabilities:{add:["SYS_PTRACE"]}} else . end' | register "$P-agent")
fi

if [ "$USE_FARGATE" = 1 ]; then
SANDBOX_SCRIPT='printf "%s" "$CA_PEM" > /tmp/ca.pem
EXECD_LISTEN=0.0.0.0:8081 EXECD_ROOT=/workspace EXECD_HOME=/home/sandbox EXECD_KEPT_JOBS_SURVIVE_IDLE=1 EXECD_LOG_FORMAT=json \
exec /usr/local/bin/execd'
TD_SANDBOX=$(jq -n --arg family "$P-sandbox" --arg exec "$(EXEC sandbox)" --argjson logs "$(LOGS sandbox)" \
  --arg img "$ECR/$P-sandbox:latest" --arg script "$SANDBOX_SCRIPT" --arg ca "$CA_PEM" \
  --arg bearer "$(param_arn sandbox-bearer)" --arg usecases "$USECASES" --argjson flogs "$(LOGS front)" --arg fimg "$ECR/$P-front:latest" --arg ek "$(param_arn e2b-key)" --arg name "$NAME" --arg btpl "${BROWSER_TEMPLATE:-}" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"256", memory:(if $usecases == "1" then "1024" else "512" end),
  executionRoleArn:$exec,
  containerDefinitions:[
    {name:"sandbox", image:$img, essential:true, entryPoint:["/bin/sh","-c"], command:[$script],
     portMappings:[{containerPort:8081}],
     environment:[{name:"CA_PEM",value:$ca},{name:"SSL_CERT_FILE",value:"/tmp/ca.pem"}],
     secrets:[{name:"EXECD_BEARER",valueFrom:$bearer}],
     logConfiguration:$logs}]}
  # The use cases'"'"' exec front, beside the sandbox: the address a coding session'"'"'s grant names. It
  # makes that session'"'"'s E2B microVM when the first request arrives (front.py).
  | if $usecases == "1" then .containerDefinitions += [
    {name:"front", image:$fimg, essential:true, portMappings:[{containerPort:8084}],
     environment:[{name:"CA_PEM",value:$ca},{name:"NAME",value:$name},{name:"E2B_BROWSER_TEMPLATE",value:$btpl}],
     secrets:[{name:"E2B_API_KEY",valueFrom:$ek}], logConfiguration:$flogs}] else . end' | register "$P-sandbox")
fi

if [ "$USECASES" = 1 ]; then
TD_STANDIN=$(jq -n --arg family "$P-standin" --arg exec "$(EXEC standin)" --argjson logs "$(LOGS standin)" \
  --arg img "$ECR/$P-standin:latest" --arg ms "$(param_arn managed-secret)" --arg ps "$(param_arn provider-secret)" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"256", memory:"512",
  executionRoleArn:$exec,
  containerDefinitions:[
    {name:"standin", image:$img, essential:true, portMappings:[{containerPort:8082}],
     secrets:[{name:"DEMO_MANAGED_SECRET",valueFrom:$ms},{name:"DEMO_PROVIDER_SECRET",valueFrom:$ps}],
     logConfiguration:$logs}]}' | register "$P-standin")
fi

# The identity provider and the broker share one task: the broker reaches Keycloak on loopback, and the
# provider's name for itself (oidc.<dns>) is what tokens say and the scripted user's browser is sent to.
# Keycloak is `start-dev` (its embedded database): a stack for a demo, torn down by down.sh.
if [ "$OIDC" = 1 ]; then
TD_OIDC=$(jq -n --arg family "$P-oidc" --arg exec "$(EXEC oidc)" --argjson logs_kc "$(LOGS keycloak)" --argjson logs_b "$(LOGS broker)" \
  --arg img "$ECR/$P-standin:latest" --arg host "oidc.$DNS" --arg realm "$(param_arn oidc-realm)" --arg ms "$(param_arn managed-secret)" \
  --arg bm "$(param_arn broker-master)" --arg ba "$(param_arn broker-admin)" --arg cs "$(param_arn oidc-client-secret)" '{
  family:$family, networkMode:"awsvpc", requiresCompatibilities:["FARGATE"], cpu:"1024", memory:"2048",
  executionRoleArn:$exec,
  containerDefinitions:[
    {name:"keycloak", image:"quay.io/keycloak/keycloak:26.4", essential:true, entryPoint:["/bin/bash","-c"],
     command:["mkdir -p /opt/keycloak/data/import && printf \"%s\" \"$REALM_JSON\" > /opt/keycloak/data/import/realm.json && exec /opt/keycloak/bin/kc.sh start-dev --import-realm --http-port=8180 --hostname=http://\($host):8180 --hostname-strict=false"],
     portMappings:[{containerPort:8180}], secrets:[{name:"REALM_JSON",valueFrom:$realm}], logConfiguration:$logs_kc},
    {name:"broker", image:$img, essential:true, entryPoint:["/usr/local/bin/beyond-ai-demo"],
     command:["oidc-endpoint","--issuer","http://\($host):8180/realms/demo","--redirect-uri","http://\($host):8085/oauth/callback"],
     portMappings:[{containerPort:8082},{containerPort:8085}],
     secrets:[{name:"DEMO_MANAGED_SECRET",valueFrom:$ms},{name:"DEMO_BROKER_MASTER",valueFrom:$bm},
       {name:"DEMO_BROKER_ADMIN",valueFrom:$ba},{name:"DEMO_OIDC_CLIENT_SECRET",valueFrom:$cs}],
     logConfiguration:$logs_b}]}' | register "$P-oidc")
fi

# ---------------------------------------------------------------------------------------------
say "services"
service() { # name, task definition arn, security group (default: the shared tasks group)
  local name="$P-$1" td=$2 sg=${3:-$SG_TASKS} cur
  local reg
  reg=$(registry "$1")
  cur=$(q ecs describe-services --cluster "$CLUSTER" --services "$name" --query 'services[?status==`ACTIVE`].taskDefinition|[0]')
  if [ -z "$cur" ]; then
    aws ecs create-service --cluster "$CLUSTER" --service-name "$name" --task-definition "$td" --desired-count 1 \
      --launch-type FARGATE --platform-version LATEST ${EXEC_FLAG:+--enable-execute-command} \
      --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$sg],assignPublicIp=ENABLED}" \
      --service-registries "registryArn=$reg" --propagate-tags SERVICE --enable-ecs-managed-tags \
      --tags "key=$TAG_KEY,value=$NAME" >/dev/null
    made "service $name"
  elif [ "$cur" != "$td" ] || [ -n "$IMAGES_CHANGED" ]; then
    aws ecs update-service --cluster "$CLUSTER" --service "$name" --task-definition "$td" --desired-count 1 \
      --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$sg],assignPublicIp=ENABLED}" \
      ${IMAGES_CHANGED:+--force-new-deployment} >/dev/null
    echo "  ~ redeploying $name"
  fi
}
# An ECS service tag cannot be added after create when the service already has none; fine here, ours
# are tagged at create.
[ "$LOAD" = 1 ] || service gateway "$TD_GATEWAY"
EXEC_FLAG=
[ "$EXECSCAN" = 0 ] || EXEC_FLAG=1
service agent "$TD_AGENT" "$SG_AGENT"
EXEC_FLAG=
SERVICES=("$P-gateway" "$P-agent")
[ "$LOAD" = 0 ] || SERVICES=("$P-agent")
if [ "$USE_FARGATE" = 1 ]; then
  service sandbox "$TD_SANDBOX" "$SG_SANDBOX"
  SERVICES+=("$P-sandbox")
fi
if [ "$USECASES" = 1 ]; then
  service standin "$TD_STANDIN"
  SERVICES+=("$P-standin")
fi
if [ "$OIDC" = 1 ]; then
  service oidc "$TD_OIDC"
  SERVICES+=("$P-oidc")
fi

if [ "$LOAD" = 1 ]; then
  service loadhost "$TD_LOADHOST"
  SERVICES+=("$P-loadhost")
fi

echo "waiting for the ${#SERVICES[@]} services to be stable..."
aws ecs wait services-stable --cluster "$CLUSTER" --services "${SERVICES[@]}"

IP=$(agent_ip)
echo
created=$(wc -l < "$WORK/made" 2>/dev/null || echo 0)
echo "up: created $created resource(s) in $REGION (tag $TAG_KEY=$NAME)."
echo "  agent   ws://$IP:$AGENT_PORT   (open to $CALLER only)"
if [ "$LOAD" = 1 ]; then
  echo "  model   scripted stand-in at loadhost.$DNS:$LOADMODEL_PORT (no gateway, no provider)"
  echo "  loadhost $(svc_ip loadhost):$LOADHOST_PORT"
else
  echo "  model   $MODEL via the gateway, provider $PROVIDER"
fi
case "$SANDBOX" in
  e2b) echo "  sandbox E2B: one microVM per session, made by demo.sh (metadata $TAG_KEY=$NAME); down.sh kills them" ;;
  both) echo "  sandbox the use cases: a Fargate workspace host, E2B microVMs (metadata $TAG_KEY=$NAME; down.sh kills them),"
    echo "          and the stand-in managed-MCP endpoint at $(svc_ip standin):$STANDIN_PORT" ;;
  load) echo "  sandbox none (the load proof: assistant-style bundles, no exec endpoint)" ;;
  oidc) echo "  sandbox none. The identity provider (Keycloak) and the broker are one task at $(svc_ip oidc):$OIDC_IDP_PORT / :$STANDIN_PORT / :$OIDC_CONTROL_PORT" ;;
  *) echo "  sandbox Fargate: one shared execd service" ;;
esac
echo "  next    ./deploy/demo/demo.sh        a scripted conversation, with a run record and report"
echo "          ./deploy/demo/demo.sh kill   kill the agent mid-job; the session survives"
[ "$USECASES" = 0 ] || echo "          ./deploy/demo/demo.sh assistant | coding | browser    the three use cases"
[ "$OIDC" = 0 ] || echo "          ./deploy/demo/demo.sh oidc   an agent acting as a signed-in user, through a managed connector"
echo "          ./deploy/demo/down.sh        delete everything, then prove it is gone"
