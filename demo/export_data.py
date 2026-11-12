"""Build `demo/data.js` for the Deal Desk page. NOT BUILT YET: runs after the final test run.

    python demo/export_data.py

Same contract as the class demos' export scripts (Lectures 5, 6, 12):

1. Rebuild everything from the raw file with the pipeline's own code (no copied numbers).
2. Fit each ranking rule at the settings model selection chose.
3. Check itself against the numbers quoted in the README and the slides, and REFUSE
   to write data.js if anything has moved (CHECKPOINTS below).
4. Write only what the page reads: one record per test-cohort project
   (fips, state, region, technology, MW, each rule's score, what actually happened)
   plus the assumptions block the page prints in its footer.

Only test-cohort projects go on the page, so every score on screen is out of sample.
"""
import sys

# Filled in from the final test run (python -m queue_risk.pipeline --final), then frozen.
CHECKPOINTS = {
    # "test projects": 3559,
    # "Gradient boosting test AUC": ...,
    # "Prec@50, gradient boosting": ...,
}


def main():
    sys.exit("demo/export_data.py is a stub until the final test evaluation (planned Nov 17). "
             "See demo/README.md for the page it will feed.")


if __name__ == "__main__":
    main()
