.PHONY: cdk-bootstrap cdk-deploy cdk-diff cdk-destroy deploy deploy-recorder deploy-engine engine-local install-engine-on-ec2 ssh-ec2 status engine-status logs engine-logs query data-summary pull-data help

# Stack name
STACK_NAME=HLRecorderStack
CDK_DIR=deploy/cdk

# --- CDK Infrastructure ---

# Bootstrap CDK (one-time per AWS account/region)
# Set CDK_DEFAULT_REGION=ap-northeast-1 and CDK_DEFAULT_ACCOUNT before running
cdk-bootstrap:
	@if [ -z "$$CDK_DEFAULT_REGION" ] || [ -z "$$CDK_DEFAULT_ACCOUNT" ]; then \
		echo "ERROR: Set CDK_DEFAULT_REGION and CDK_DEFAULT_ACCOUNT"; \
		echo "Example: CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-bootstrap"; \
		exit 1; \
	fi
	cd $(CDK_DIR) && cdk bootstrap

# Deploy infrastructure (creates EC2 + EBS, starts recording automatically)
cdk-deploy:
	@if [ -z "$$CDK_DEFAULT_REGION" ] || [ -z "$$CDK_DEFAULT_ACCOUNT" ]; then \
		echo "ERROR: Set CDK_DEFAULT_REGION and CDK_DEFAULT_ACCOUNT"; \
		echo "Example: CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-deploy"; \
		exit 1; \
	fi
	cd $(CDK_DIR) && cdk deploy --require-approval broadening

# Preview infrastructure changes
cdk-diff:
	cd $(CDK_DIR) && cdk diff

# Destroy infrastructure (WARNING: will snapshot EBS but delete EC2)
cdk-destroy:
	cd $(CDK_DIR) && cdk destroy

# --- Deployment ---

# Deploy code changes to EC2 (git pull + restart service(s))
# Default: restarts both recorder + engine. Use deploy-recorder or
# deploy-engine to scope to one service.
deploy:
	./scripts/deploy.sh

deploy-recorder:
	./scripts/deploy.sh --service recorder

# Restarting the engine triggers its §5.5 restart-drift gate. If any
# ghost/orphan/position-mismatch fires, the engine writes
# data/engine/restart_blocked and the scanner stays suspended until you SSH
# in, investigate, and `rm` the flag.
deploy-engine:
	./scripts/deploy.sh --service engine

# Install or refresh /etc/systemd/system/hl-engine.service on the live box.
# Use this after pushing a new branch that changes the unit content or after
# rotating SSM secrets. Pulls latest code first (so the unit's ExecStartPre
# can find the up-to-date scripts/fetch-engine-secrets.sh), then runs the
# installer as root via SSM. After this, `make deploy` works.
install-engine-on-ec2:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	BRANCH=$$(git rev-parse --abbrev-ref HEAD) && \
	echo "==> Installing hl-engine.service on $$INSTANCE_ID from branch $$BRANCH..." && \
	CMD_ID=$$(aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters '{"commands":[ \
			"set -xe", \
			"cd /opt/hl-recorder", \
			"sudo -u ec2-user git fetch origin", \
			"sudo -u ec2-user git checkout '"$$BRANCH"'", \
			"sudo -u ec2-user git reset --hard origin/'"$$BRANCH"'", \
			"chmod +x /opt/hl-recorder/scripts/fetch-engine-secrets.sh /opt/hl-recorder/scripts/install-engine-systemd.sh", \
			"bash /opt/hl-recorder/scripts/install-engine-systemd.sh" \
		]}' \
		--query "Command.CommandId" --output text) && \
	echo "  Command ID: $$CMD_ID" && \
	sleep 8 && \
	aws ssm get-command-invocation --command-id "$$CMD_ID" --instance-id "$$INSTANCE_ID" \
		--query "[Status, StandardOutputContent, StandardErrorContent]" --output text

# Run the engine locally against your dev .env.local. Loads env vars
# (HL_*, TG_*) from .env.local, then invokes hl-engine with the in-repo
# config files. paper_mode=true (the strategy.yaml default) means no real
# orders fire. Ctrl-C for graceful shutdown.
engine-local:
	@if [ ! -f .env.local ]; then \
		echo "ERROR: .env.local not found. Copy from .env.local.example:"; \
		echo "  cp .env.local.example .env.local && chmod 600 .env.local"; \
		exit 1; \
	fi
	@if [ "$$(stat -f '%A' .env.local 2>/dev/null || stat -c '%a' .env.local)" != "600" ]; then \
		echo "WARN: .env.local is not 0600 — fixing"; \
		chmod 600 .env.local; \
	fi
	set -a; . ./.env.local; set +a; \
	uv run hl-engine \
		--strategy-config config/strategy.yaml \
		--deploy-config config/deploy.yaml \
		--symbols-config config/symbols.yaml \
		--log-level INFO

# --- Remote Access & Monitoring ---

# SSH to EC2 via SSM (no SSH key needed)
ssh-ec2:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	echo "Starting SSM session to $$INSTANCE_ID..." && \
	aws ssm start-session --target $$INSTANCE_ID

# Check recorder status
status:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	echo "Fetching recorder status from $$INSTANCE_ID..." && \
	aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters 'commands=["systemctl status hl-recorder.service --no-pager", "echo", "tail -20 /data/logs/recorder.log"]' \
		--query "Command.CommandId" \
		--output text | xargs -I {} sh -c \
		'sleep 3 && aws ssm get-command-invocation --command-id {} --instance-id '"$$INSTANCE_ID"' --query "StandardOutputContent" --output text'

# Quick data summary (counts by venue/event)
data-summary:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID."; exit 1; \
	fi && \
	echo "Querying data summary from $$INSTANCE_ID..." && \
	CMD_ID=$$(aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters 'commands=["/usr/local/bin/duckdb -c \"SELECT venue, event, COUNT(*) FROM read_parquet('"'"'/data/**/*.parquet'"'"') GROUP BY venue, event ORDER BY 3 DESC\""]' \
		--query "Command.CommandId" --output text) && \
	sleep 5 && \
	aws ssm get-command-invocation --command-id "$$CMD_ID" --instance-id "$$INSTANCE_ID" \
		--query "StandardOutputContent" --output text

# Run a custom DuckDB query on the instance: make query Q="SELECT ..."
query:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID."; exit 1; \
	fi && \
	if [ -z "$$Q" ]; then \
		echo 'Usage: make query Q="SELECT venue, COUNT(*) FROM read_parquet('"'"'/data/**/*.parquet'"'"') GROUP BY venue"'; exit 1; \
	fi && \
	CMD_ID=$$(aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters "commands=[\"/usr/local/bin/duckdb -c \\\"$$Q\\\"\"]" \
		--query "Command.CommandId" --output text) && \
	sleep 5 && \
	aws ssm get-command-invocation --command-id "$$CMD_ID" --instance-id "$$INSTANCE_ID" \
		--query "StandardOutputContent" --output text

# Engine status: systemd state + last 30 journal lines + restart-drift flag
engine-status:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	echo "Fetching engine status from $$INSTANCE_ID..." && \
	aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters 'commands=["systemctl status hl-engine.service --no-pager", "echo", "echo === restart_blocked flag ===", "ls -la /data/engine/restart_blocked 2>/dev/null || echo none", "echo", "echo === halt flag ===", "ls -la /data/engine/halt 2>/dev/null || echo none", "echo", "echo === recent journal ===", "journalctl -u hl-engine.service -n 30 --no-pager"]' \
		--query "Command.CommandId" \
		--output text | xargs -I {} sh -c \
		'sleep 3 && aws ssm get-command-invocation --command-id {} --instance-id '"$$INSTANCE_ID"' --query "StandardOutputContent" --output text'

# Tail engine journal
engine-logs:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	echo "Fetching engine logs from $$INSTANCE_ID..." && \
	aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters 'commands=["journalctl -u hl-engine.service -n 200 --no-pager"]' \
		--query "Command.CommandId" \
		--output text | xargs -I {} sh -c \
		'sleep 3 && aws ssm get-command-invocation --command-id {} --instance-id '"$$INSTANCE_ID"' --query "StandardOutputContent" --output text'

# Tail logs from the recorder
logs:
	@INSTANCE_ID=$$(aws cloudformation describe-stacks --stack-name $(STACK_NAME) \
		--query "Stacks[0].Outputs[?ExportName=='HLRecorderInstanceID'].OutputValue" \
		--output text) && \
	if [ -z "$$INSTANCE_ID" ]; then \
		echo "ERROR: Could not fetch instance ID. Is the stack deployed?"; \
		exit 1; \
	fi && \
	echo "Fetching logs from $$INSTANCE_ID..." && \
	aws ssm send-command \
		--instance-ids "$$INSTANCE_ID" \
		--document-name "AWS-RunShellScript" \
		--parameters 'commands=["tail -100 /data/logs/recorder.log"]' \
		--query "Command.CommandId" \
		--output text | xargs -I {} sh -c \
		'sleep 3 && aws ssm get-command-invocation --command-id {} --instance-id '"$$INSTANCE_ID"' --query "StandardOutputContent" --output text'

# Pull all archived parquet data from S3 to local ./data/
pull-data:
	@./scripts/pull-data.sh

# Help
help:
	@echo "HLAnalysis Recorder - Deployment Commands"
	@echo ""
	@echo "Infrastructure (one-time setup):"
	@echo "  cdk-bootstrap     Bootstrap CDK in Tokyo (set CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=<account>)"
	@echo "  cdk-deploy        Deploy EC2 + EBS infrastructure (starts recording automatically)"
	@echo "  cdk-diff          Preview infrastructure changes"
	@echo "  cdk-destroy       Tear down infrastructure (WARNING: destroys EC2, snapshots EBS)"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy            Deploy code + restart BOTH recorder and engine"
	@echo "  deploy-recorder   Deploy code + restart recorder only"
	@echo "  deploy-engine     Deploy code + restart engine only (triggers §5.5 drift gate)"
	@echo "  install-engine-on-ec2  (Re)install /etc/systemd/system/hl-engine.service on the live box"
	@echo ""
	@echo "Local dev:"
	@echo "  engine-local      Run engine on this machine in paper mode (loads .env.local)"
	@echo ""
	@echo "Monitoring:"
	@echo "  ssh-ec2           Start SSM session to EC2 instance"
	@echo "  status            Check recorder service status + last 20 log lines"
	@echo "  logs              Tail last 100 lines of recorder logs"
	@echo "  engine-status     Check engine service + restart_blocked / halt flags + journal"
	@echo "  engine-logs       Tail last 200 lines of engine journal"
	@echo "  data-summary      Show event counts grouped by venue and event type"
	@echo "  query Q=\"...\"     Run custom DuckDB query, e.g. Q=\"SELECT COUNT(*) FROM ...\""
	@echo "  pull-data         Sync archived data from S3 to local ./data/ (incremental)"
	@echo ""
	@echo "Example workflow:"
	@echo "  1. CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-bootstrap"
	@echo "  2. CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-deploy"
	@echo "  3. make status"
	@echo "  4. make logs"
	@echo "  5. (later) make deploy"
