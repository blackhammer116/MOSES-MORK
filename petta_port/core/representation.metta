;; =========================================
;; Core Data Representations for MOSES-MORK
;; ported to MeTTa (PeTTa compatible)
;; =========================================

;; -----------------------------------------
;; Hyperparameters representation
;; -----------------------------------------
;; Defined functionally. To retrieve, evaluate (hyperparam <key>).
(= (hyperparam mutation_rate) 0.3)
(= (hyperparam crossover_rate) 0.5)
(= (hyperparam num_generations) 30)
(= (hyperparam neighborhood_size) 50)
(= (hyperparam fg_type) "beta")
(= (hyperparam max_iter) 100)
(= (hyperparam bernoulli_prob) 0.5)
(= (hyperparam uniform_prob) 0.5)
(= (hyperparam initial_population_size) 2)
(= (hyperparam exemplar_selection_size) 7)
(= (hyperparam min_crossover_neighbors) 5)
(= (hyperparam evidence_propagation_steps) 30)
(= (hyperparam max_dist) 50)
(= (hyperparam feature_order) 4)

;; -----------------------------------------
;; Instance Representation
;; -----------------------------------------
;; In MeTTa, instead of strings, we use native expression trees:
;; (Instance <expr> <id> <score>)
;; Example: (Instance (AND A B) 1 0.8)

;; Helpers to extract data from an Instance
(= (get-score (Instance $expr $id $score)) $score)
(= (get-expr (Instance $expr $id $score)) $expr)
(= (get-id (Instance $expr $id $score)) $id)

;; Instance constructor (default score 0.0)
(= (init-instance $expr $id) (Instance $expr $id 0.0))

;; Complexity (Weakness implementation): count tokens/size of tree.
;; Base cases for count-nodes
(= (count-nodes Nil) 0)
(= (count-nodes $sym) (if (== (get-type $sym) Symbol) 1 0))

;; For expression trees, we can count recursively if needed,
;; or just defer to a Python python utility if complex structure.

;; -----------------------------------------
;; Deme Representation
;; -----------------------------------------
;; (Deme <id> <generation> <Instances Tuple>)
;; Examples: (Deme "01" 0 ((Instance (AND A B) 1 0.8)))

(= (get-deme-generation (Deme $id $gen $instances)) $gen)
(= (get-deme-instances (Deme $id $gen $instances)) $instances)

(= (increment-generation (Deme $id $gen $instances))
   (Deme $id (+ $gen 1) $instances))

(= (set-deme-instances (Deme $id $gen $old_instances) $new_instances)
   (Deme $id $gen $new_instances))

;; -----------------------------------------
;; Truth Tables & Knobs Representation
;; -----------------------------------------
;; Knobs in Python held evaluation state implicitly.
;; In MeTTa, we load them as Atoms in the Space:
;; (TruthValue A (True False ...))
;; (TruthValue target (True True ...))
;; Then evaluating a variable purely queries the Space.
