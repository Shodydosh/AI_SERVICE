"""Tạo training data với positive pairs rõ ràng cho fine-tuning."""
import json
from pathlib import Path

# Tạo training data với positive pairs chính xác
training_data = {
    "candidate_texts": [
        "Kỹ sư phần mềm với 5 năm kinh nghiệm Python, FastAPI, PostgreSQL",
        "Nhà khoa học dữ liệu với nền tảng Machine Learning, TensorFlow, PyTorch",
        "Lập trình viên Backend với 3 năm kinh nghiệm microservices, REST API",
        "Lập trình viên Frontend với React, TypeScript, Vue.js",
        "DevOps Engineer với Kubernetes, Docker, CI/CD pipelines",
        "Full-stack Developer với Node.js, React, MongoDB",
        "Data Engineer với Spark, Hadoop, Airflow",
        "Mobile Developer với React Native, Flutter",
        "QA Engineer với Selenium, Cypress, automation testing",
        "Product Manager với kinh nghiệm quản lý sản phẩm công nghệ"
    ],
    "job_texts": [
        "Senior Python Developer - FastAPI, PostgreSQL, 5+ năm kinh nghiệm",
        "ML Engineer - TensorFlow, PyTorch, nghiên cứu và phát triển",
        "Backend Engineer - Microservices, REST APIs, 3+ năm kinh nghiệm",
        "React Developer - TypeScript, modern frontend frameworks",
        "DevOps Engineer - Kubernetes, Docker, CI/CD pipelines",
        "Full-stack Developer - Node.js, React, MongoDB, 2+ năm",
        "Data Engineer - Spark, Hadoop, Airflow, ETL pipelines",
        "Mobile Developer - React Native, Flutter, iOS/Android",
        "QA Engineer - Selenium, Cypress, test automation",
        "Product Manager - Quản lý sản phẩm công nghệ, Agile",
        "Java Developer - Spring Boot, Microservices",
        "Cloud Architect - AWS, Azure, GCP",
        "Security Engineer - Penetration testing, security audit",
        "Blockchain Developer - Solidity, Ethereum, Smart contracts",
        "AI Researcher - Deep Learning, Computer Vision"
    ],
    # Positive pairs: (candidate_idx, job_idx)
    "train_pairs": [
        [0, 0],  # Python Developer -> Senior Python Developer
        [1, 1],  # Data Scientist -> ML Engineer
        [1, 14], # Data Scientist -> AI Researcher (cũng phù hợp)
        [2, 2],  # Backend Developer -> Backend Engineer
        [3, 3],  # Frontend Developer -> React Developer
        [4, 4],  # DevOps -> DevOps Engineer
        [5, 5],  # Full-stack -> Full-stack Developer
        [6, 6],  # Data Engineer -> Data Engineer
        [7, 7],  # Mobile Developer -> Mobile Developer
        [8, 8],  # QA Engineer -> QA Engineer
        [9, 9],  # Product Manager -> Product Manager
        # Thêm một số pairs phù hợp khác
        [0, 2],  # Python Developer có thể làm Backend
        [2, 0],  # Backend có thể làm Python
        [3, 7],  # Frontend có thể làm Mobile (React Native)
        [5, 3],  # Full-stack có thể làm Frontend
        [5, 2],  # Full-stack có thể làm Backend
    ],
    "val_pairs": [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4]
    ]
}

# Lưu training data
output_file = Path("data/training_data_improved.json")
output_file.parent.mkdir(exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"✓ Training data created: {output_file}")
print(f"  - Candidates: {len(training_data['candidate_texts'])}")
print(f"  - Jobs: {len(training_data['job_texts'])}")
print(f"  - Positive pairs: {len(training_data['train_pairs'])}")
print(f"  - Validation pairs: {len(training_data['val_pairs'])}")

