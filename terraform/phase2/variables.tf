variable "aws_region" {
  description = "PHẢI trùng region với Phase 1 (cùng VPC) — mặc định giống terraform/phase1"
  type        = string
  default     = "ap-southeast-1"
}

variable "project_name" {
  type    = string
  default = "ecommerce-mlops"
}

variable "my_ip" {
  description = "IP public của bạn (CIDR, vd 1.2.3.4/32) — giới hạn SSH vào 2 GPU instance"
  type        = string
}

variable "key_pair_name" {
  description = "Tên EC2 key-pair — dùng lại đúng key-pair đã tạo cho Phase 1 hoặc key khác tùy bạn"
  type        = string
}

variable "gpu_instance_type" {
  description = "g4dn.xlarge — 1x NVIDIA T4 16GB. Qwen2.5-7B-Instruct-AWQ (~4-5GB weights) vừa thoải mái; 7B FP16 (~14GB) sẽ KHÔNG đủ chỗ cho KV-cache trên GPU này"
  type        = string
  default     = "g4dn.xlarge"
}

variable "vllm_model" {
  description = "7B AWQ thay vì 3B FP16 — 3B đã được note trong graph.py là không đủ tin cậy cho tool-calling; AWQ giữ VRAM nhẹ hơn cả 3B FP16 (~4-5GB vs ~6GB) trong khi chất lượng cao hơn hẳn (mất ~1-2% do quantize, không đáng kể so với chênh lệch quy mô model)"
  type        = string
  default     = "Qwen/Qwen2.5-7B-Instruct-AWQ"
}

variable "vllm_quantization" {
  description = "Cờ --quantization cho vLLM khi model là bản AWQ/GPTQ — set thành chuỗi rỗng \"\" (KHÔNG phải null) nếu dùng model FP16 thường"
  type        = string
  default     = "awq"
}

variable "spot_max_price" {
  description = "Giá tối đa sẵn sàng trả cho Spot (USD/giờ) — để trống dùng giá On-Demand làm trần mặc định của AWS"
  type        = string
  default     = null
}

variable "model_s3_prefix" {
  description = <<-EOT
    Prefix trong S3 bucket (dùng lại bucket Phase 1) chứa model weights đã tải sẵn
    từ HuggingFace — PHẢI tự upload 1 lần TRƯỚC khi apply (không tự động hóa được):
      huggingface-cli download Qwen/Qwen2.5-7B-Instruct-AWQ --local-dir ./qwen2.5-7b-instruct-awq
      aws s3 sync ./qwen2.5-7b-instruct-awq s3://<bucket-tu-phase1>/models/qwen2.5-7b-instruct-awq/
    Vì sao không tải trực tiếp từ HuggingFace lúc container start: tránh phụ thuộc
    mạng ra ngoài VPC + rate-limit HuggingFace lúc 2 instance cùng tải song song.
  EOT
  type        = string
  default     = "models/qwen2.5-7b-instruct-awq"
}

variable "gpu_subnet_id_override" {
  description = <<-EOT
    Ghi đè subnet_id thay vì lấy từ Phase 1 output (chỉ có 1 subnet, 1 AZ).
    Dùng khi AZ mặc định (Phase 1) gặp sự cố AWS-side (vd RunInstances trả
    500 liên tục do control-plane/capacity cục bộ 1 AZ) — đổi sang AZ khác
    trong cùng VPC để né, không cần đụng gì tới Phase 1 đang chạy thật.
    Để trống ("") thì dùng subnet mặc định từ Phase 1.
  EOT
  type        = string
  default     = ""
}

variable "use_spot" {
  description = <<-EOT
    true = Spot (rẻ, có thể bị AWS thu hồi bất cứ lúc nào). false = On-Demand
    (đắt hơn ~2x nhưng đảm bảo có capacity ngay, không bị reclaim) — dùng tạm
    khi Spot capacity căng (RunInstances trả 500 liên tục / bị reclaim ngay
    sau khi cấp, gặp thật ở ap-southeast-1 sáng nay).
  EOT
  type        = bool
  default     = true
}
