"""Print a short-term Amazon Bedrock API key minted from the local AWS credential chain.

The live suite runs this when AWS_BEARER_TOKEN_BEDROCK is unset, so no long-lived Bedrock secret
is stored: the token is a SigV4-presigned request (valid up to 12 hours), created locally, and
nothing is created in the AWS account. Exits non-zero, printing nothing, when no credential exists.
"""
import os
import sys

try:
    from aws_bedrock_token_generator import provide_token

    print(provide_token(region=os.environ.get("VERIFY_BEDROCK_REGION", "us-east-1")))
except Exception:  # noqa: BLE001 - no credential means no Bedrock cells, not a failure
    sys.exit(1)
