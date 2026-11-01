#!/bin/bash

echo "🔬 Comprehensive Multi-Agent System Evaluation"
echo "============================================="

# Default parameters
DATASET="itops_dataset.jsonl"
TOTAL_REQUESTS=200
USERS=5
OUTPUT_DIR="./results"
QUICK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --total-requests)
      TOTAL_REQUESTS="$2"
      shift 2
      ;;
    --users)
      USERS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --quick)
      QUICK_MODE=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "🎯 Multi-Agent System Evaluation"
      echo ""
      echo "Options:"
      echo "  --dataset FILE         JSONL dataset (default: itops_dataset.jsonl)"
      echo "  --total-requests N     Total requests for evaluation (default: 200)"
      echo "  --users N             Concurrent users (default: 5)"
      echo "  --output DIR          Output directory (default: ./results)"
      echo "  --quick               Quick mode with fewer requests (default: false)"
      echo "  --help                Show this help"
      echo ""
      echo "Examples:"
      echo "  $0                                    # Full evaluation (200 requests)"
      echo "  $0 --quick                            # Quick evaluation"
      echo "  $0 --total-requests 500 --users 10   # Large-scale evaluation"
      echo "  $0 --dataset my_dataset.jsonl        # Custom dataset"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Quick mode adjustments
if [ "$QUICK_MODE" = true ]; then
    TOTAL_REQUESTS=60
    USERS=3
    echo "🚀 Quick evaluation mode enabled"
fi

echo "📋 Evaluation Configuration:"
echo "   Dataset: $DATASET"
echo "   Total requests: $TOTAL_REQUESTS"
echo "   Concurrent users: $USERS"
echo "   Output: $OUTPUT_DIR"
echo "   Mode: $([ "$QUICK_MODE" = true ] && echo "Quick" || echo "Full")"
echo ""

# Validate dataset
if [ ! -f "$DATASET" ]; then
    echo ""
    echo "❌ Dataset not found: $DATASET"
    echo ""
    echo "Available JSONL files:"
    find . -name "*.jsonl" -type f 2>/dev/null || echo "   No JSONL files found"
    echo ""
    echo "Expected format: {\"task_type\": \"...\", \"utterance\": \"...\", \"graphql_query\": \"...\"}"
    exit 1
fi

# Quick dataset validation
echo ""
echo "🔍 Validating dataset..."
TOTAL_LINES=$(wc -l < "$DATASET" 2>/dev/null || echo "0")

# Get valid lines count and category breakdown
VALIDATION_OUTPUT=$(python3 -c "
import json
count = 0
categories = {'security': 0, 'performance': 0, 'visualization': 0, 'general': 0}
try:
    with open('$DATASET', 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'utterance' in data:
                    count += 1
                    utterance = data.get('utterance', '').lower()
                    if any(term in utterance for term in ['vulnerability', 'cve', 'security', 'risk']):
                        categories['security'] += 1
                    elif any(term in utterance for term in ['performance', 'latency', 'optimization']):
                        categories['performance'] += 1
                    elif any(term in utterance for term in ['visualize', 'chart', 'dashboard']):
                        categories['visualization'] += 1
                    else:
                        categories['general'] += 1
            except:
                pass
    print(f'{count}')
    print(f'Security={categories[\"security\"]},Performance={categories[\"performance\"]},Visualization={categories[\"visualization\"]},General={categories[\"general\"]}')
except:
    print('0')
    print('Security=0,Performance=0,Visualization=0,General=0')
" 2>/dev/null)

VALID_LINES=$(echo "$VALIDATION_OUTPUT" | head -1)
CATEGORY_INFO=$(echo "$VALIDATION_OUTPUT" | tail -1)

if [ "$VALID_LINES" -eq 0 ]; then
    echo "❌ No valid entries found in $DATASET"
    exit 1
fi

echo "✅ Dataset validation:"
echo "   Total lines: $TOTAL_LINES"
echo "   Valid entries: $VALID_LINES"
echo "   Categories: $CATEGORY_INFO"

if [ "$VALID_LINES" -lt "$TOTAL_REQUESTS" ]; then
    echo "⚠️  Total requests ($TOTAL_REQUESTS) > available entries ($VALID_LINES)"
    echo "   Evaluation will use all available entries with cycling"
fi

# Check dependencies
echo ""
echo "🔍 Checking system requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi

# Check Python packages
python3 -c "import pandas, httpx, asyncio" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Installing required packages..."
    pip install pandas httpx 2>/dev/null || {
        echo "❌ Failed to install packages. Please run:"
        echo "   pip install pandas httpx"
        exit 1
    }
fi

# Check orchestrator
echo "🔍 Checking orchestrator..."
if curl -s http://localhost:8000/health > /dev/null 2>&1 || curl -s http://localhost:8000 > /dev/null 2>&1; then
    echo "✅ Orchestrator is running at http://localhost:8000"
else
    echo "❌ Orchestrator not reachable"
    echo "   Please start your agents first: ./start_all_agents.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Show estimated time
ESTIMATED_TIME=$((TOTAL_REQUESTS * 3 / 60))  # Rough estimate: 3 seconds per request
echo ""
echo "⏱️  Estimated time: ~${ESTIMATED_TIME} minutes"

# Final confirmation for longer runs
if [ "$TOTAL_REQUESTS" -gt 100 ] && [ "$QUICK_MODE" != true ]; then
    echo ""
    echo "⚠️  This evaluation will:"
    echo "   • Send $TOTAL_REQUESTS requests"
    echo "   • Take approximately $ESTIMATED_TIME minutes to complete"
    echo "   • Test all agents with $USERS concurrent users"
    echo ""
    read -p "Continue with evaluation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Evaluation cancelled. Use --quick for a faster test."
        exit 0
    fi
fi

echo ""
echo "🚀 Starting Multi-Agent System Evaluation..."
echo ""

# Run the evaluation
START_TIME=$(date +%s)

# Use consistent output directory (no timestamps)
FINAL_OUTPUT_DIR="$OUTPUT_DIR"
mkdir -p "$FINAL_OUTPUT_DIR"

# Build the correct arguments for eval.py
PYTHON_ARGS="--dataset $DATASET --total-requests $TOTAL_REQUESTS --users $USERS --output-dir $FINAL_OUTPUT_DIR"

if [ "$QUICK_MODE" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --quick"
fi

echo "🔄 Running evaluation with: python3 eval.py $PYTHON_ARGS"
echo ""

python3 eval.py $PYTHON_ARGS

EVAL_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

echo ""
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "   Total time: ${DURATION_MIN}m ${DURATION_SEC}s"
    
    if [ -d "$FINAL_OUTPUT_DIR" ]; then
        echo ""
        echo "📊 Generated Files:"
        find "$FINAL_OUTPUT_DIR" -type f | sort | while read file; do
            size=$(du -h "$file" | cut -f1)
            basename_file=$(basename "$file")
            case "$basename_file" in
                *.csv)
                    echo "   📋 $basename_file ($size) - Detailed evaluation data"
                    ;;
                *.json)
                    echo "   📈 $basename_file ($size) - Statistical summaries"
                    ;;
                *.tex)
                    echo "   📄 $basename_file ($size) - LaTeX tables"
                    ;;
                *.md)
                    echo "   📝 $basename_file ($size) - Summary report"
                    ;;
                *)
                    echo "   📄 $basename_file ($size)"
                    ;;
            esac
        done
        
        # Show quick results preview if stats file exists
        STATS_FILE=$(find "$FINAL_OUTPUT_DIR" -name "*.json" | head -1)
        if [ -f "$STATS_FILE" ]; then
            echo ""
            echo "📈 Quick Results Preview:"
            python3 -c "
import json
try:
    with open('$STATS_FILE', 'r') as f:
        data = json.load(f)
    
    # Try different possible structures
    if 'overall' in data:
        overall = data['overall']
        print(f'   📊 Overall Success Rate: {overall.get(\"success_rate\", 0):.1%}')
        print(f'   ⏱️  Avg Response Time: {overall.get(\"avg_response_time\", 0):.3f}s')
        if 'avg_evaluation_score' in overall:
            print(f'   🎯 Avg Score: {overall.get(\"avg_evaluation_score\", 0):.1f}/10')
    else:
        # Try to extract basic stats
        for key, value in data.items():
            if isinstance(value, dict) and 'success_rate' in value:
                print(f'   {key}: {value.get(\"success_rate\", 0):.1%} success')
                break
        
except Exception as e:
    print(f'   Could not parse results: {e}')
"
        fi
        
        echo ""
        echo "📁 All results saved to: $FINAL_OUTPUT_DIR"
        
    else
        echo "⚠️  Results directory not found"
    fi
    
else
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "   Duration: ${DURATION_MIN}m ${DURATION_SEC}s"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check if all agents are running properly"
    echo "   2. Verify dataset format is correct"
    echo "   3. Ensure sufficient system resources"
    echo "   4. Try --quick mode for debugging"
    
    if [ -d "$FINAL_OUTPUT_DIR" ]; then
        echo "   5. Check partial results in: $FINAL_OUTPUT_DIR"
    fi
fi

echo ""
echo "🏁 Evaluation completed."
