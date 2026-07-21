output "vllm_replica_public_ips" {
  description = "Public IP của 2 replica — dùng để SSH debug trực tiếp nếu cần"
  value       = aws_instance.vllm_replica[*].public_ip
}

output "vllm_replica_private_ips" {
  description = "Private IP của 2 replica — dùng trong nginx upstream (Instance A gọi qua private IP, cùng VPC, không tính egress cost)"
  value       = aws_instance.vllm_replica[*].private_ip
}

output "nginx_upstream_snippet" {
  description = "Dán trực tiếp vào /etc/nginx/sites-available/vllm-lb trên Instance A — xem AWS_VLLM_DEPLOY.md mục 4"
  value       = <<-EOT
    upstream vllm_backend {
        server ${aws_instance.vllm_replica[0].private_ip}:8000;
        server ${aws_instance.vllm_replica[1].private_ip}:8000;
    }
  EOT
}

output "ssh_commands" {
  value = [
    "ssh -i your-key.pem ubuntu@${aws_instance.vllm_replica[0].public_ip}  # Replica B",
    "ssh -i your-key.pem ubuntu@${aws_instance.vllm_replica[1].public_ip}  # Replica C",
  ]
}
