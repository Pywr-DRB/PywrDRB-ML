#!/bin/bash

# Enable debugging if needed
# set -x

# Get job ID from command line argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <job_id>"
    echo "Example: $0 134989"
    exit 1
fi

jobid="$1"
echo "Processing all folders containing job ID: $jobid"

# Set epsilon values and base output directory
epsilon="0.01,0.01,0.01"
outputsDir="../outputs"

# Set the path to the CLI executable
MOEAFramework5Path="../../MOEAFramework-5.0"
cliPath="$MOEAFramework5Path/cli"

# Set MOEAFramework download info
MOEAFrameworkURL="https://github.com/MOEAFramework/MOEAFramework/releases/download/v5.0/MOEAFramework-5.0.tar.gz"
MOEAFrameworkTar="MOEAFramework-5.0.tar.gz"

# Check if MOEAFramework directory exists
if [ ! -d "$MOEAFramework5Path" ]; then
    echo "MOEAFramework-5.0 not found. Downloading..."

    # Download using curl or wget
    curl -L -o "$MOEAFrameworkTar" "$MOEAFrameworkURL"

    # Extract using tar
    tar -xzf "$MOEAFrameworkTar" -C ../../

    # Clean up
    rm "$MOEAFrameworkTar"
fi

# Check the permission is given
if [ ! -x "$cliPath" ]; then
    echo "Error: CLI at $cliPath is not executable. Run:"
    echo "chmod +x $cliPath"
    exit 1
fi

# Create metrics directory if it doesn't exist
mkdir -p "$metricDir"

# Find all directories containing the job ID
matchingDirs=()
for dir in "$outputsDir"/*/; do
    # Extract directory name without path
    dirName=$(basename "$dir")
    if [[ "$dirName" == *"$jobid"* ]]; then
        matchingDirs+=("$dir")
    fi
done

# Check if we found any matching directories
if [ ${#matchingDirs[@]} -eq 0 ]; then
    echo "Error: No directories found containing job ID '$jobid' in $outputsDir"
    exit 1
fi

echo "Found ${#matchingDirs[@]} directories containing job ID '$jobid':"
for dir in "${matchingDirs[@]}"; do
    echo "  - $(basename "$dir")"
done
echo ""

# Process each matching directory
for setDir in "${matchingDirs[@]}"; do
    echo "Processing directory: $(basename "$setDir")"
    
    # Set up paths for this directory
    runtimeDir="${setDir}runtimes"
    refFile="${setDir}borg.ref"
    metricDir="${setDir}metrics"
    
    # Create metrics directory if it doesn't exist
    mkdir -p "$metricDir"

    # Step 1: Merge all .set files in the data directory
    fileList=()
    for f in "$setDir"/*.set; do
        # Check if the file actually exists (in case glob doesn't match anything)
        if [ -f "$f" ]; then
            fileList+=("$f")
        fi
    done

    # Check if we found any .set files
    if [ ${#fileList[@]} -eq 0 ]; then
        echo "  Warning: No .set files found in $setDir"
        continue
    fi

    echo "  Found ${#fileList[@]} .set files to merge"
    "$cliPath" ResultFileMerger --epsilon "$epsilon" --output "$refFile" "${fileList[@]}"

    # Step 2: Evaluate metrics for runtime files
    runtimeFound=false
    for f in "$runtimeDir"/*.runtime; do
        # Check if the file actually exists (in case glob doesn't match anything)
        if [ ! -f "$f" ]; then
            continue
        fi
        
        runtimeFound=true
        filename=$(basename "$f" .runtime)
        outfile="$metricDir/$filename.metric"

        echo "  Processing runtime file: $(basename "$f")"
        
        # Check if file contains '# Version=5'
        if ! grep -q "# Version=5" "$f"; then
            echo "  Adding header to $(basename "$f")"
            grep "^#" "$refFile" > temp_header.txt
            cat "$f" >> temp_header.txt
            mv temp_header.txt "$f"
        fi

        "$cliPath" MetricsEvaluator --epsilon "$epsilon" --input "$f" --output "$outfile" --reference "$refFile"
    done

    if [ "$runtimeFound" = false ]; then
        echo "  Warning: No .runtime files found in $runtimeDir"
    fi
    
    echo "  Completed processing $(basename "$setDir")"
    echo ""
done

echo "All directories processed successfully!"