rule quasar:
    output:
        "src/tex/figures/quasar.pdf"
    script:
        "src/scripts/quasar.py"

rule transit:
    output:
        "src/tex/figures/transit.pdf"
        "src/tex/figures/transit_posteriors.pdf"
    script:
        "src/scripts/transit.py"
