"""
Test external link filter
"""
import re

def filter_external_links(text: str) -> str:
    """Remove external links from response, keep only gamatrain.com links."""
    import re
    
    # Pattern 1: Match full URLs (http://... or https://...)
    # But NOT gamatrain.com
    external_url_pattern1 = r'https?://(?!(?:www\.)?gamatrain\.com)[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}[^\s\)]*'
    
    # Pattern 2: Match www.example.com (without http)
    external_url_pattern2 = r'www\.(?!gamatrain\.com)[a-zA-Z0-9][-a-zA-Z0-9.]*\.[a-zA-Z]{2,}[^\s\)]*'
    
    # Remove external URLs (replace with empty string, preserving spaces)
    cleaned_text = re.sub(external_url_pattern1, '', text)
    cleaned_text = re.sub(external_url_pattern2, '', cleaned_text)
    
    # DON'T clean up spaces - that was causing the problem!
    # Just return as-is
    return cleaned_text


# Test cases
test_cases = [
    {
        "input": "Check out https://www.sciencedirect.com/article for more info",
        "expected": "Check out  for more info"  # Note: double space preserved
    },
    {
        "input": "Visit https://gamatrain.com/blog/test for our blog",
        "expected": "Visit https://gamatrain.com/blog/test for our blog"
    },
    {
        "input": "See www.nature.com and https://britannica.com",
        "expected": "See  and "  # Spaces preserved
    },
    {
        "input": "Photosynthesis is a process. See https://example.com for details.",
        "expected": "Photosynthesis is a process. See  for details."  # Space preserved
    },
    {
        "input": "Our site: https://gamatrain.com/blog/11597/photosynthesis and external: https://example.com",
        "expected": "Our site: https://gamatrain.com/blog/11597/photosynthesis and external: "
    }
]

print("=" * 70)
print("Testing External Link Filter")
print("=" * 70)

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    result = filter_external_links(test["input"])
    expected = test["expected"]
    
    # Compare directly (spaces matter now!)
    if result == expected:
        print(f"\n✅ Test {i} PASSED")
        passed += 1
    else:
        print(f"\n❌ Test {i} FAILED")
        print(f"   Input:    {test['input'][:80]}...")
        print(f"   Expected: '{expected}'")
        print(f"   Got:      '{result}'")
        failed += 1

print("\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 70)
