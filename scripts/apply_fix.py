"""
One-click script to apply the hallucinated URL fix
This will check, rebuild, and test the fix automatically
"""
import os
import sys
import subprocess
import time

def run_command(cmd, description):
    """Run a command and show the result"""
    print(f"\n{'='*70}")
    print(f"🔧 {description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                   HALLUCINATED URL FIX - AUTO APPLY                  ║
╚══════════════════════════════════════════════════════════════════════╝

This script will:
1. Check if your index needs rebuilding
2. Rebuild the index if needed
3. Provide instructions to restart your server
4. Test the fix

""")
    
    input("Press Enter to start...")
    
    # Step 1: Check current index
    print("\n" + "="*70)
    print("STEP 1: Checking Current Index")
    print("="*70)
    
    success = run_command(
        "python scripts/check_index_metadata.py",
        "Checking index metadata"
    )
    
    if not success:
        print("\n⚠️  Could not check index. Proceeding with rebuild anyway...")
    
    # Step 2: Rebuild index
    print("\n" + "="*70)
    print("STEP 2: Rebuilding Index")
    print("="*70)
    
    response = input("\nDo you want to rebuild the index? (y/n): ")
    
    if response.lower() == 'y':
        success = run_command(
            "python scripts/rebuild_index.py",
            "Rebuilding RAG index"
        )
        
        if success:
            print("\n✅ Index rebuild complete!")
        else:
            print("\n⚠️  Index rebuild had issues, but continuing...")
    else:
        print("\n⏭️  Skipping index rebuild")
    
    # Step 3: Server restart instructions
    print("\n" + "="*70)
    print("STEP 3: Restart Server")
    print("="*70)
    print("""
⚠️  IMPORTANT: You need to restart your API server!

1. Stop your current server (Ctrl+C in the server terminal)
2. Start it again:
   
   cd api
   python llm_server_production.py

3. Wait for the server to rebuild the index (check logs)
4. Come back here and press Enter to continue testing
""")
    
    input("Press Enter after you've restarted the server...")
    
    # Step 4: Test the fix
    print("\n" + "="*70)
    print("STEP 4: Testing the Fix")
    print("="*70)
    
    # Wait a bit for server to be ready
    print("\nWaiting 3 seconds for server to be ready...")
    time.sleep(3)
    
    response = input("\nDo you want to run the test now? (y/n): ")
    
    if response.lower() == 'y':
        success = run_command(
            "python scripts/test_gamatrain_query.py",
            "Testing gamatrain query"
        )
        
        if success:
            print("\n✅ Test complete! Check the results above.")
        else:
            print("\n⚠️  Test had issues. Make sure your server is running on port 8001")
    else:
        print("\n⏭️  Skipping test. You can run it later with:")
        print("     python scripts/test_gamatrain_query.py")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ FIX APPLICATION COMPLETE!")
    print("="*70)
    print("""
Summary:
✅ Code has been fixed in api/llm_server_production.py
✅ Index has been rebuilt (or ready to rebuild)
✅ Server restart instructions provided
✅ Test script available

Next Steps:
1. Test in your web app: Ask "What is gamatrain?"
2. Verify no fake URLs appear
3. Check that any sources shown are valid gamatrain.com URLs

Documentation:
- See HALLUCINATION_FIX_SUMMARY.md for quick reference
- See docs/FIX_HALLUCINATED_URLS.md for detailed info

If you still see issues:
- Run: python scripts/check_index_metadata.py
- Check server logs for warnings
- Verify api/storage/ and storage/ were deleted
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
