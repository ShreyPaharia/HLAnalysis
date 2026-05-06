package main

import (
	"os"

	"github.com/aws/aws-cdk-go/awscdk/v2"
	"github.com/aws/jsii-runtime-go"
)

func main() {
	defer jsii.Close()

	app := awscdk.NewApp(nil)

	enableSSH := os.Getenv("ENABLE_SSH") == "true"
	keyPair := os.Getenv("EC2_KEY_PAIR")
	sshCIDR := os.Getenv("SSH_CIDR")
	if enableSSH && sshCIDR == "" {
		panic("SSH_CIDR is required when ENABLE_SSH=true (e.g. SSH_CIDR=203.0.113.5/32)")
	}

	NewHLRecorderStack(app, "HLRecorderStack", &HLRecorderStackProps{
		StackProps: awscdk.StackProps{
			Description: jsii.String("HLAnalysis recorder EC2 deployment in Tokyo with SSM access"),
			Env:         env(),
		},
		EnableSSH: enableSSH,
		KeyPair:   keyPair,
		SSHCIDR:   sshCIDR,
	})

	app.Synth(nil)
}

func env() *awscdk.Environment {
	account := os.Getenv("CDK_DEFAULT_ACCOUNT")
	region := os.Getenv("CDK_DEFAULT_REGION")

	if account == "" || region == "" {
		return nil
	}

	return &awscdk.Environment{
		Account: jsii.String(account),
		Region:  jsii.String(region),
	}
}
