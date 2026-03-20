# Create src directory structure
import os

# Create src directory
if not os.path.exists('src'):
    os.makedirs('src')

print("Created src/ directory")
print("\nProject structure:")
print("├── main.py (main application)")
print("├── requirements.txt")
print("└── src/")
print("    ├── __init__.py")
print("    ├── fetchers.py (paper fetching)")  
print("    ├── summarizer.py (AI summaries)")
print("    ├── clustering.py (paper clustering)")
print("    └── utils.py (utilities)")

print("\nTo set up:")
print("1. Create the src/ directory")
print("2. Move the src_*.py files into src/ and rename them:")
print("   - src_fetchers.py → src/fetchers.py")
print("   - src_summarizer.py → src/summarizer.py") 
print("   - src_clustering.py → src/clustering.py")
print("   - src_utils.py → src/utils.py")
print("   - src___init__.py → src/__init__.py")
print("3. pip install -r requirements.txt")
print("4. streamlit run main.py")