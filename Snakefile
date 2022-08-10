rule quasar:
    output:
        "src/tex/figures/quasar.pdf",
        "src/tex/figures/quasar_posteriors.pdf"
    script:
        "src/scripts/quasar.py"

rule quasar2:
    output:
        "src/tex/figures/quasar2.pdf",
        "src/tex/figures/quasar2_posteriors.pdf"
    script:
        "src/scripts/quasar2.py"

rule transit:
    output:
        "src/tex/figures/transit.pdf",
        "src/tex/figures/transit_posteriors.pdf"
    script:
        "src/scripts/transit.py"
