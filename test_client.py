import requests
import json

BASE_URL = "http://localhost:5000"

def test_encode():
    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium."
    ]
    
    print("\n1. Testing basic encoding...")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": sentences}
    )
    
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        print("Embeddings generated successfully!")
        return embeddings
    else:
        print(f"Error in encoding: {response.json()}")
        return None

def test_similarity(embeddings, threshold=0.0):
    if embeddings is None:
        return
    
    print(f"\n2. Testing similarity with threshold={threshold}...")
    response = requests.post(
        f"{BASE_URL}/similarity",
        json={
            "embeddings": embeddings,
            "threshold": threshold
        }
    )
    
    if response.status_code == 200:
        similarities = response.json()["similarities"]
        print("Similarities calculated successfully!")
        print("\nSimilarity matrix:")
        for row in similarities:
            print([f"{val:.4f}" for val in row])
    else:
        print(f"Error in similarity calculation: {response.json()}")

def test_semantic_search():
    query = "What's the weather like?"
    documents = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
        "The forecast predicts rain tomorrow.",
        "I love sunny days at the beach."
    ]
    
    print("\n3. Testing semantic search...")
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": query,
            "documents": documents,
            "top_k": 3,
            "threshold": 0.5
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nQuery: {results['query']}")
        print("\nTop results:")
        for result in results['results']:
            print(f"Similarity: {result['similarity']:.4f} - {result['document']}")
    else:
        print(f"Error in semantic search: {response.json()}")

def test_batch_processing():
    batches = [
        {
            "sentences": ["The weather is nice.", "It's raining outside."]
        },
        {
            "sentences": ["I love programming.", "Coding is fun."]
        }
    ]
    
    print("\n4. Testing batch processing...")
    response = requests.post(
        f"{BASE_URL}/batch_process",
        json={"batches": batches}
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"\nProcessed {results['total_batches']} batches")
        for i, batch_result in enumerate(results['results']):
            print(f"\nBatch {i+1} processing time: {batch_result['processing_time']:.4f}s")
            print("Similarity matrix:")
            for row in batch_result['similarities']:
                print([f"{val:.4f}" for val in row])
    else:
        print(f"Error in batch processing: {response.json()}")

def test_product_search():
    # Product documents database
    products = [
        "Apple iPhone 13 Pro Max 256GB - Space Gray - Unlocked",
        "Samsung Galaxy S21 Ultra 5G 256GB - Phantom Black - Unlocked",
        "Sony WH-1000XM4 Wireless Noise Cancelling Overhead Headphones",
        "Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones",
        "Apple MacBook Pro 14-inch M1 Pro 16GB RAM 512GB SSD",
        "Dell XPS 13 9310 Touchscreen Laptop - 11th Gen Intel Core i7",
        "Samsung 65-inch QLED 4K UHD Smart TV with Alexa Built-in",
        "LG 55-inch OLED 4K UHD Smart TV with AI ThinQ",
        "Apple AirPods Pro with MagSafe Charging Case",
        "Samsung Galaxy Buds Pro True Wireless Earbuds"
    ]
    
    # Test queries
    queries = [
        "Best noise cancelling headphones",
        "Latest iPhone model",
        "High-end laptop for professionals",
        "Large screen smart TV",
        "Wireless earbuds with good sound"
    ]
    
    print("\nTesting product search with different queries...")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "documents": products,
                "top_k": 3,
                "threshold": 0.3  # Lower threshold to catch more relevant products
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\nTop matching products:")
            for result in results['results']:
                print(f"Similarity: {result['similarity']:.4f} - {result['document']}")
        else:
            print(f"Error in search: {response.json()}")
        
        print("\n" + "-"*80)

def test_product_similarity():
    # Group similar products
    phones = [
        "Apple iPhone 13 Pro Max 256GB - Space Gray - Unlocked",
        "Samsung Galaxy S21 Ultra 5G 256GB - Phantom Black - Unlocked"
    ]
    
    headphones = [
        "Sony WH-1000XM4 Wireless Noise Cancelling Overhead Headphones",
        "Bose QuietComfort 45 Bluetooth Wireless Noise Cancelling Headphones"
    ]
    
    laptops = [
        "Apple MacBook Pro 14-inch M1 Pro 16GB RAM 512GB SSD",
        "Dell XPS 13 9310 Touchscreen Laptop - 11th Gen Intel Core i7"
    ]
    
    print("\nTesting similarity between similar products...")
    
    # Test phone similarity
    print("\n1. Testing phone similarity:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": phones}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test headphone similarity
    print("\n2. Testing headphone similarity:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": headphones}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test laptop similarity
    print("\n3. Testing laptop similarity:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": laptops}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)

def test_faq_search():
    # FAQ database with questions and answers
    faqs = [
        {
            "question": "How do I reset my password?",
            "answer": "To reset your password, go to the login page and click 'Forgot Password'. Enter your email address and follow the instructions sent to your inbox."
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept all major credit cards (Visa, MasterCard, American Express), PayPal, and bank transfers."
        },
        {
            "question": "How can I track my order?",
            "answer": "You can track your order by logging into your account and clicking on 'Order History'. You'll find tracking information for all your recent orders."
        },
        {
            "question": "What is your return policy?",
            "answer": "We offer a 30-day return policy. Items must be unused and in their original packaging. Contact our support team to initiate a return."
        },
        {
            "question": "How do I contact customer support?",
            "answer": "You can reach our customer support team 24/7 through live chat, email at support@example.com, or by calling 1-800-123-4567."
        },
        {
            "question": "Do you offer international shipping?",
            "answer": "Yes, we ship to over 100 countries worldwide. Shipping costs and delivery times vary by destination."
        },
        {
            "question": "How can I update my account information?",
            "answer": "Log into your account, go to 'Account Settings', and click 'Edit Profile' to update your personal information."
        },
        {
            "question": "What are your business hours?",
            "answer": "Our customer service is available Monday through Friday, 9 AM to 6 PM EST. Online orders can be placed 24/7."
        }
    ]
    
    # Test user questions
    user_questions = [
        "I forgot my password, what should I do?",
        "Can I pay with PayPal?",
        "Where is my package?",
        "I want to return something",
        "Need help with my account",
        "Do you ship to Canada?",
        "How to change my email address",
        "When can I call support?"
    ]
    
    print("\nTesting FAQ search with user questions...")
    
    # Extract questions and answers
    faq_questions = [faq["question"] for faq in faqs]
    faq_answers = [faq["answer"] for faq in faqs]
    
    for user_question in user_questions:
        print(f"\nUser Question: '{user_question}'")
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": user_question,
                "documents": faq_questions,
                "top_k": 3,
                "threshold": 0.3
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\nTop matching FAQs:")
            for result in results['results']:
                idx = faq_questions.index(result['document'])
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"FAQ Question: {result['document']}")
                print(f"FAQ Answer: {faq_answers[idx]}")
                print()
        else:
            print(f"Error in search: {response.json()}")
        
        print("\n" + "-"*80)

def test_faq_similarity():
    # Group similar FAQ questions
    account_questions = [
        "How do I reset my password?",
        "How can I update my account information?",
        "How do I change my email address?"
    ]
    
    shipping_questions = [
        "How can I track my order?",
        "Do you offer international shipping?",
        "What are your shipping rates?"
    ]
    
    support_questions = [
        "How do I contact customer support?",
        "What are your business hours?",
        "When can I reach support?"
    ]
    
    print("\nTesting similarity between FAQ categories...")
    
    # Test account questions similarity
    print("\n1. Testing account-related questions:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": account_questions}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test shipping questions similarity
    print("\n2. Testing shipping-related questions:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": shipping_questions}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test support questions similarity
    print("\n3. Testing support-related questions:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": support_questions}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)

def test_diverse_search():
    # Diverse document collection
    documents = [
        # Technical documents
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning algorithms can learn patterns from data without explicit programming.",
        "The transformer architecture revolutionized natural language processing tasks.",
        
        # News articles
        "Global temperatures have risen by 1.1°C since the pre-industrial era.",
        "The stock market reached record highs amid positive economic indicators.",
        "Scientists discovered a new species of deep-sea creatures in the Pacific Ocean.",
        
        # Literary excerpts
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness.",
        "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole.",
        
        # Scientific concepts
        "Quantum entanglement describes a phenomenon where particles remain connected across distances.",
        "The theory of relativity fundamentally changed our understanding of space and time.",
        "DNA contains the genetic instructions used in the development and functioning of all living organisms.",
        
        # Historical facts
        "The Industrial Revolution began in Britain in the late 18th century.",
        "The Renaissance period marked a cultural rebirth in Europe from the 14th to 17th centuries.",
        "The first human landing on the Moon occurred on July 20, 1969."
    ]
    
    # Diverse test queries
    queries = [
        # Technical queries
        "What is Python programming?",
        "How do machine learning models work?",
        "Explain transformer models in NLP",
        
        # News-related queries
        "Latest climate change statistics",
        "Current stock market performance",
        "Recent scientific discoveries",
        
        # Literary queries
        "Famous opening lines in literature",
        "Classic novel beginnings",
        "Tolkien's writing style",
        
        # Scientific queries
        "Quantum physics concepts",
        "Einstein's theories explained",
        "Genetic material in cells",
        
        # Historical queries
        "Industrial Revolution origins",
        "Renaissance period significance",
        "Moon landing details"
    ]
    
    print("\nTesting diverse search with different types of content...")
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "documents": documents,
                "top_k": 3,
                "threshold": 0.3
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\nTop matching documents:")
            for result in results['results']:
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Document: {result['document']}")
                print()
        else:
            print(f"Error in search: {response.json()}")
        
        print("\n" + "-"*80)

def test_cross_domain_similarity():
    # Documents from different domains
    tech_docs = [
        "Artificial intelligence is transforming industries worldwide.",
        "Blockchain technology enables secure decentralized transactions.",
        "Cloud computing provides scalable computing resources on demand."
    ]
    
    business_docs = [
        "Market analysis shows growing demand for sustainable products.",
        "Digital transformation is essential for business competitiveness.",
        "Customer experience drives brand loyalty and revenue growth."
    ]
    
    science_docs = [
        "Climate change impacts are becoming increasingly evident.",
        "Renewable energy technologies are advancing rapidly.",
        "Biodiversity conservation is crucial for ecosystem stability."
    ]
    
    print("\nTesting similarity across different domains...")
    
    # Test tech document similarity
    print("\n1. Testing technical documents:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": tech_docs}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test business document similarity
    print("\n2. Testing business documents:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": business_docs}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test science document similarity
    print("\n3. Testing science documents:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": science_docs}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)

def test_build_failures():
    # Build failure logs and their resolutions
    build_issues = [
        {
            "error_log": """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! 
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@18.2.0
npm ERR! node_modules/react
npm ERR!   react@"^18.2.0" from the root project
npm ERR! 
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^16.8.0 || ^17.0.0" from react-scripts@4.0.3
npm ERR! node_modules/react-scripts
npm ERR!   react-scripts@"^4.0.3" from the root project
            """,
            "resolution": """
1. Check your package.json for React version
2. Either downgrade React to version 17:
   npm install react@17.0.2 react-dom@17.0.2
3. Or upgrade react-scripts to a version compatible with React 18:
   npm install react-scripts@5.0.1
4. Run npm install again
            """
        },
        {
            "error_log": """
FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':app:compileDebugJavaWithJavac'.
> java.lang.OutOfMemoryError: Java heap space
            """,
            "resolution": """
1. Add the following to your gradle.properties:
   org.gradle.jvmargs=-Xmx2048m -XX:MaxPermSize=512m -XX:+HeapDumpOnOutOfMemoryError
2. If using Android Studio, go to Help -> Edit Custom VM Options
3. Add: -Xmx2048m
4. Restart Android Studio and rebuild
            """
        },
        {
            "error_log": """
npm ERR! code ENOENT
npm ERR! syscall open
npm ERR! path /Users/user/project/package.json
npm ERR! errno -2
npm ERR! enoent ENOENT: no such file or directory, open '/Users/user/project/package.json'
            """,
            "resolution": """
1. Make sure you're in the correct project directory
2. Run: npm init -y to create a new package.json
3. Install your dependencies again
4. Run npm install
            """
        },
        {
            "error_log": """
FAILURE: Build failed with an exception.
* What went wrong:
Could not determine the dependencies of task ':app:compileDebugJavaWithJavac'.
> Could not resolve all dependencies for configuration ':app:debugCompileClasspath'.
> Could not find com.android.support:appcompat-v7:28.0.0.
            """,
            "resolution": """
1. Update your build.gradle file:
   implementation 'androidx.appcompat:appcompat:1.3.0'
2. Make sure you're using AndroidX dependencies
3. Sync your project with Gradle files
4. Clean and rebuild the project
            """
        },
        {
            "error_log": """
npm ERR! code ELIFECYCLE
npm ERR! errno 1
npm ERR! my-app@1.0.0 start: `react-scripts start`
npm ERR! Exit status 1
npm ERR! Failed at the my-app@1.0.0 start script.
            """,
            "resolution": """
1. Delete node_modules folder
2. Delete package-lock.json
3. Clear npm cache: npm cache clean --force
4. Run npm install
5. Try starting the app again: npm start
            """
        }
    ]
    
    # Test error queries
    error_queries = [
        "npm dependency resolution error",
        "Java heap space error in Gradle",
        "package.json not found",
        "Android support library missing",
        "npm start script failed"
    ]
    
    print("\nTesting build failure resolution search...")
    
    # Extract error logs and resolutions
    error_logs = [issue["error_log"] for issue in build_issues]
    resolutions = [issue["resolution"] for issue in build_issues]
    
    for query in error_queries:
        print(f"\nQuery: '{query}'")
        response = requests.post(
            f"{BASE_URL}/search",
            json={
                "query": query,
                "documents": error_logs,
                "top_k": 3,
                "threshold": 0.2  # Lower threshold for error logs
            }
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\nTop matching error logs and resolutions:")
            for result in results['results']:
                idx = error_logs.index(result['document'])
                print(f"\nSimilarity: {result['similarity']:.4f}")
                print("Error Log:")
                print(result['document'].strip())
                print("\nResolution:")
                print(resolutions[idx].strip())
                print("\n" + "-"*80)
        else:
            print(f"Error in search: {response.json()}")

def test_error_similarity():
    # Group similar error types
    npm_errors = [
        "npm ERR! code ERESOLVE",
        "npm ERR! code ENOENT",
        "npm ERR! code ELIFECYCLE"
    ]
    
    gradle_errors = [
        "Execution failed for task ':app:compileDebugJavaWithJavac'",
        "Could not determine the dependencies of task",
        "Could not resolve all dependencies for configuration"
    ]
    
    print("\nTesting similarity between error types...")
    
    # Test npm error similarity
    print("\n1. Testing npm errors:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": npm_errors}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)
    
    # Test gradle error similarity
    print("\n2. Testing gradle errors:")
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"sentences": gradle_errors}
    )
    if response.status_code == 200:
        embeddings = response.json()["embeddings"]
        test_similarity(embeddings, threshold=0.0)

def test_devops_resolution():
    # Test error logs
    error_logs = [
        {
            "error_log": """
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! While resolving: my-app@1.0.0
npm ERR! Found: react@18.2.0
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^16.8.0 || ^17.0.0" from react-scripts@4.0.3
            """,
            "expected_type": "npm"
        },
        {
            "error_log": """
FAILURE: Build failed with an exception.
* What went wrong:
Execution failed for task ':app:compileDebugJavaWithJavac'.
> java.lang.OutOfMemoryError: Java heap space
            """,
            "expected_type": "gradle"
        },
        {
            "error_log": """
ERROR: failed to solve: failed to compute cache key: failed to calculate checksum
            """,
            "expected_type": "docker"
        },
        {
            "error_log": """
Error from server (NotFound): deployments.apps not found
            """,
            "expected_type": "kubernetes"
        }
    ]
    
    print("\nTesting DevOps Error Resolution System...")
    
    # Test error classification
    print("\n1. Testing Error Classification:")
    for error in error_logs:
        print(f"\nError Log: {error['error_log'].strip()}")
        response = requests.post(
            f"{BASE_URL}/classify_error",
            json={"error_log": error["error_log"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Classified as: {result['error_type']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("Type scores:")
            for type_name, score in result['scores'].items():
                print(f"  {type_name}: {score:.4f}")
        else:
            print(f"Error in classification: {response.json()}")
    
    # Test solution finding
    print("\n2. Testing Solution Finding:")
    for error in error_logs:
        print(f"\nError Log: {error['error_log'].strip()}")
        response = requests.post(
            f"{BASE_URL}/find_solution",
            json={"error_log": error["error_log"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Error Type: {result['error_type']}")
            print("\nTop Solutions:")
            for solution in result['solutions']:
                print(f"\nSimilarity: {solution['similarity']:.4f}")
                print(f"Severity: {solution['severity']}")
                print(f"Frequency: {solution['frequency']}")
                print("Resolution Steps:")
                print(solution['resolution'])
        else:
            print(f"Error in solution finding: {response.json()}")
    
    # Test trend analysis
    print("\n3. Testing Trend Analysis:")
    response = requests.post(
        f"{BASE_URL}/analyze_trends",
        json={"error_logs": [e["error_log"] for e in error_logs]}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\nError Distribution:")
        for error_type, count in result['error_distribution'].items():
            print(f"  {error_type}: {count}")
        
        print("\nSeverity Distribution:")
        for severity, count in result['severity_distribution'].items():
            print(f"  {severity}: {count}")
        
        print(f"\nAverage Resolution Time: {result['average_resolution_time']:.2f} minutes")
        
        print("\nMost Common Errors:")
        for error_type, count in result['common_errors']:
            print(f"  {error_type}: {count}")
    else:
        print(f"Error in trend analysis: {response.json()}")

if __name__ == "__main__":
    print("Testing diverse search and similarity...")
    
    # Test diverse search with different types of content
    test_diverse_search()
    
    # Test similarity across different domains
    test_cross_domain_similarity()
    
    print("Testing build failure resolution search...")
    
    # Test build failure search
    test_build_failures()
    
    # Test error type similarity
    test_error_similarity()
    
    print("Testing DevOps Error Resolution System...")
    test_devops_resolution() 