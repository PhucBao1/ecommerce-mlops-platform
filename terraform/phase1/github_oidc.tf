# GitHub Actions OIDC — no long-lived AWS keys stored in GitHub Secrets
#
# How it works:
#   1. GitHub generates a short-lived JWT per workflow run
#   2. GitHub Actions assumes this IAM role via OIDC federation
#   3. Role grants ECR push + SSM deploy permissions
#   4. No AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY stored anywhere
#
# Usage in workflow:
#   permissions:
#     id-token: write
#     contents: read
#   - uses: aws-actions/configure-aws-credentials@v4
#     with:
#       role-to-assume: ${{ vars.AWS_ROLE_ARN }}   # output from terraform
#       aws-region: ap-southeast-1

resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  # GitHub's OIDC thumbprint — stable, no need to rotate
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "${var.project_name}-github-actions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
      Action    = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringLike = {
          # Giới hạn đúng repo + branch (bản gốc dùng ":*" cho phép MỌI ref — quá rộng)
          "token.actions.githubusercontent.com:sub" = "repo:${var.github_repo}:ref:refs/heads/${var.github_branch}"
        }
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

# ECR: login + push all 3 repos
resource "aws_iam_role_policy" "github_ecr" {
  name = "ecr-push"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = [
          aws_ecr_repository.recsys_api.arn,
          aws_ecr_repository.sentiment_api.arn,
          aws_ecr_repository.agent_api.arn,
          aws_ecr_repository.recsys_consumer.arn,
          aws_ecr_repository.recsys_producer.arn,
        ]
      }
    ]
  })
}

# SSM: trigger rolling deploy on EC2 after image push
resource "aws_iam_role_policy" "github_ssm" {
  name = "ssm-deploy"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:ListCommandInvocations",
      ]
      Resource = [
        "arn:aws:ec2:${var.aws_region}:*:instance/*",
        "arn:aws:ssm:${var.aws_region}::document/AWS-RunShellScript",
        "arn:aws:ssm:${var.aws_region}:*:*",
      ]
    }]
  })
}
