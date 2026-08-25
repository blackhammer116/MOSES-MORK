;; =========================================
;; Main Entry Point for MOSES-MORK PeTTa Port
;; =========================================

;; Import Python loader for data
!(import! &self petta_port.core.loaders)

;; Load Truth Table CSV directly into MORK Space
!(petta_port.core.loaders.load_data_and_inject "example_data/test_parity_4.csv" &self "O")

;; Import core MeTTa files
!(import! &self petta_port/core/representation.metta)
!(import! &self petta_port/core/fitness.metta)
!(import! &self petta_port/variation/operators.metta)
!(import! &self petta_port/eda/factor_graph.metta)
!(import! &self petta_port/moses/loop.metta)

;; Initialize the starting instance
!(add-atom &self (Instance (AND A B) 1 0.0))

;; Trigger the evaluation of the starting instance
!(score-instance (Instance (AND A B) 1 0.0))

;; Trigger the main loop execution
!(moses-loop (Instance (AND A B) 1 0.0) 1 1)
