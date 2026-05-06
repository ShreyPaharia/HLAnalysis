.PHONY: cdk-bootstrap cdk-deploy cdk-diff cdk-destroy deploy ssh-ec2 status logs query data-summary help

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

# Deploy code changes to EC2 (git pull + restart service)
deploy:
	./scripts/deploy.sh

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
	@echo "  deploy            Deploy code changes (git pull + restart service on EC2)"
	@echo ""
	@echo "Monitoring:"
	@echo "  ssh-ec2           Start SSM session to EC2 instance"
	@echo "  status            Check recorder service status + last 20 log lines"
	@echo "  logs              Tail last 100 lines of recorder logs"
	@echo "  data-summary      Show event counts grouped by venue and event type"
	@echo "  query Q=\"...\"     Run custom DuckDB query, e.g. Q=\"SELECT COUNT(*) FROM ...\""
	@echo ""
	@echo "Example workflow:"
	@echo "  1. CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-bootstrap"
	@echo "  2. CDK_DEFAULT_REGION=ap-northeast-1 CDK_DEFAULT_ACCOUNT=473317027471 make cdk-deploy"
	@echo "  3. make status"
	@echo "  4. make logs"
	@echo "  5. (later) make deploy"
