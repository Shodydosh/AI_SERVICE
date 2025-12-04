"""Test script để test candidate creation API với 10 samples."""
import requests
import json
import time
from typing import List, Dict

# API base URL
BASE_URL = "http://localhost:8000/api/v1"

# 10 Sample candidates
SAMPLE_CANDIDATES = [
    {
        "candidate_id": "TEST_001",
        "title": "Nhân Viên Kế Toán",
        "skills": "Excel, Kế toán, Báo cáo tài chính, SAP, QuickBooks",
        "experience": "5 năm kinh nghiệm làm kế toán tại công ty lớn, xử lý báo cáo tài chính hàng tháng, quản lý sổ sách kế toán",
        "name": "Nguyễn Văn A",
        "email": "nguyenvana@example.com"
    },
    {
        "candidate_id": "TEST_002",
        "title": "Lập Trình Viên Python",
        "skills": "Python, Django, Flask, PostgreSQL, REST API, Git",
        "experience": "3 năm kinh nghiệm phát triển web application với Python, làm việc với Django framework, xây dựng REST API",
        "name": "Trần Thị B",
        "email": "tranthib@example.com"
    },
    {
        "candidate_id": "TEST_003",
        "title": "Nhân Viên Marketing",
        "skills": "Digital Marketing, SEO, Google Ads, Facebook Ads, Content Writing",
        "experience": "4 năm kinh nghiệm marketing online, quản lý chiến dịch quảng cáo Google và Facebook, tối ưu SEO",
        "name": "Lê Văn C",
        "email": "levanc@example.com"
    },
    {
        "candidate_id": "TEST_004",
        "title": "Kỹ Sư Phần Mềm",
        "skills": "Java, Spring Boot, Microservices, Docker, Kubernetes, AWS",
        "experience": "6 năm kinh nghiệm phát triển phần mềm enterprise, xây dựng hệ thống microservices, làm việc với cloud AWS",
        "name": "Phạm Thị D",
        "email": "phamthid@example.com"
    },
    {
        "candidate_id": "TEST_005",
        "title": "Nhân Viên Nhân Sự",
        "skills": "Tuyển dụng, Quản lý nhân sự, HRIS, Đào tạo, Quan hệ lao động",
        "experience": "4 năm kinh nghiệm quản lý nhân sự, tuyển dụng nhân viên, tổ chức đào tạo, xử lý các vấn đề quan hệ lao động",
        "name": "Hoàng Văn E",
        "email": "hoangvane@example.com"
    },
    {
        "candidate_id": "TEST_006",
        "title": "Data Analyst",
        "skills": "SQL, Python, Pandas, Tableau, Power BI, Excel, Data Visualization",
        "experience": "3 năm kinh nghiệm phân tích dữ liệu, xây dựng dashboard với Tableau và Power BI, viết query SQL phức tạp",
        "name": "Vũ Thị F",
        "email": "vuthif@example.com"
    },
    {
        "candidate_id": "TEST_007",
        "title": "Frontend Developer",
        "skills": "React, JavaScript, TypeScript, HTML, CSS, Redux, Next.js",
        "experience": "4 năm kinh nghiệm phát triển frontend, làm việc với React và TypeScript, xây dựng responsive web applications",
        "name": "Đặng Văn G",
        "email": "dangvang@example.com"
    },
    {
        "candidate_id": "TEST_008",
        "title": "Kế Toán Trưởng",
        "skills": "Kế toán tài chính, Báo cáo tài chính, Thuế, Kiểm toán, Quản lý tài chính",
        "experience": "8 năm kinh nghiệm kế toán, 3 năm làm kế toán trưởng, quản lý đội ngũ kế toán, lập báo cáo tài chính",
        "name": "Bùi Thị H",
        "email": "buithih@example.com"
    },
    {
        "candidate_id": "TEST_009",
        "title": "DevOps Engineer",
        "skills": "Docker, Kubernetes, CI/CD, Jenkins, GitLab, AWS, Terraform, Ansible",
        "experience": "5 năm kinh nghiệm DevOps, xây dựng CI/CD pipeline, quản lý infrastructure trên cloud, tự động hóa deployment",
        "name": "Ngô Văn I",
        "email": "ngovani@example.com"
    },
    {
        "candidate_id": "TEST_010",
        "title": "Product Manager",
        "skills": "Product Management, Agile, Scrum, User Research, Product Strategy, Analytics",
        "experience": "6 năm kinh nghiệm quản lý sản phẩm, làm việc với Agile/Scrum, nghiên cứu người dùng, xây dựng chiến lược sản phẩm",
        "name": "Đỗ Thị K",
        "email": "dothik@example.com"
    }
]


def test_health_check():
    """Test health check endpoint."""
    print("\n" + "="*80)
    print("TEST 1: Health Check")
    print("="*80)
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_create_single_candidate(candidate: Dict):
    """Test tạo một candidate."""
    print("\n" + "="*80)
    print(f"TEST: Create Single Candidate - {candidate['candidate_id']}")
    print("="*80)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/candidates",
            json=candidate,
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('message', '')}")
            print(f"   Candidate ID: {data.get('data', {}).get('candidate_id')}")
            print(f"   Embeddings Generated: {data.get('data', {}).get('embeddings_generated')}")
            print(f"   Recommendations Pending: {data.get('data', {}).get('recommendations_pending')}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_create_batch_candidates(candidates: List[Dict]):
    """Test tạo batch candidates."""
    print("\n" + "="*80)
    print(f"TEST: Create Batch Candidates ({len(candidates)} candidates)")
    print("="*80)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/candidates/batch",
            json={"candidates": candidates},
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Time: {elapsed_time:.2f}s")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success: {data.get('message', '')}")
            print(f"   Total Requested: {data.get('data', {}).get('total_requested')}")
            print(f"   Created: {data.get('data', {}).get('created')}")
            print(f"   Failed: {data.get('data', {}).get('failed')}")
            
            if data.get('data', {}).get('failed_candidates'):
                print(f"   Failed Candidates: {data.get('data', {}).get('failed_candidates')}")
            
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_get_candidate(candidate_id: str):
    """Test lấy thông tin candidate."""
    print(f"\nTEST: Get Candidate {candidate_id}")
    
    try:
        response = requests.get(f"{BASE_URL}/candidates/{candidate_id}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found: {data.get('name')} ({data.get('candidate_id')})")
            return True
        else:
            print(f"⚠️  Not found or error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_get_recommendations(candidate_id: str):
    """Test lấy recommendations cho candidate."""
    print(f"\nTEST: Get Recommendations for {candidate_id}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/multi-filter/recommend/jobs",
            json={
                "candidate_id": candidate_id,
                "limit": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data.get('recommendations', [])
            print(f"✅ Found {len(recommendations)} recommendations")
            
            if recommendations:
                print("   Top 3 jobs:")
                for i, job in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {job.get('title')} - {job.get('company', 'N/A')}")
            
            return True
        else:
            print(f"⚠️  Error: {response.status_code} - {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def wait_for_background_tasks(seconds: int = 15):
    """Đợi background tasks hoàn thành."""
    print(f"\n⏳ Waiting {seconds} seconds for background tasks to complete...")
    time.sleep(seconds)
    print("✅ Continue testing...")


def main():
    """Main test function."""
    print("\n" + "="*80)
    print("🧪 TEST CANDIDATE CREATION API - 10 SAMPLES")
    print("="*80)
    
    results = {
        "health_check": False,
        "single_create": [],
        "batch_create": False,
        "get_candidates": [],
        "get_recommendations": []
    }
    
    # Test 1: Health Check
    results["health_check"] = test_health_check()
    if not results["health_check"]:
        print("\n❌ Health check failed. Please make sure API is running.")
        print("   Start API with: python main.py")
        return
    
    # Test 2: Create single candidates (first 3)
    print("\n" + "="*80)
    print("TEST 2: Create Single Candidates (First 3)")
    print("="*80)
    
    for candidate in SAMPLE_CANDIDATES[:3]:
        success = test_create_single_candidate(candidate)
        results["single_create"].append(success)
        time.sleep(1)  # Small delay between requests
    
    # Test 3: Create batch (remaining 7)
    print("\n" + "="*80)
    print("TEST 3: Create Batch Candidates (Remaining 7)")
    print("="*80)
    
    results["batch_create"] = test_create_batch_candidates(SAMPLE_CANDIDATES[3:])
    
    # Wait for background tasks
    wait_for_background_tasks(15)
    
    # Test 4: Get candidates
    print("\n" + "="*80)
    print("TEST 4: Get Candidates")
    print("="*80)
    
    for candidate in SAMPLE_CANDIDATES:
        success = test_get_candidate(candidate["candidate_id"])
        results["get_candidates"].append(success)
    
    # Test 5: Get recommendations
    print("\n" + "="*80)
    print("TEST 5: Get Recommendations")
    print("="*80)
    
    for candidate in SAMPLE_CANDIDATES[:5]:  # Test first 5
        success = test_get_recommendations(candidate["candidate_id"])
        results["get_recommendations"].append(success)
        time.sleep(1)
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    print(f"Health Check: {'✅' if results['health_check'] else '❌'}")
    print(f"Single Create: {sum(results['single_create'])}/{len(results['single_create'])} ✅")
    print(f"Batch Create: {'✅' if results['batch_create'] else '❌'}")
    print(f"Get Candidates: {sum(results['get_candidates'])}/{len(results['get_candidates'])} ✅")
    print(f"Get Recommendations: {sum(results['get_recommendations'])}/{len(results['get_recommendations'])} ✅")
    
    total_tests = (
        1 +  # health check
        len(results['single_create']) +
        1 +  # batch create
        len(results['get_candidates']) +
        len(results['get_recommendations'])
    )
    
    passed_tests = (
        (1 if results['health_check'] else 0) +
        sum(results['single_create']) +
        (1 if results['batch_create'] else 0) +
        sum(results['get_candidates']) +
        sum(results['get_recommendations'])
    )
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")


if __name__ == "__main__":
    main()

