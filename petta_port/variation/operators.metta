;; =========================================
;; Variation Quantale (Operators)
;; Crossover and Mutation natively in MeTTa
;; =========================================

;; Load Python random module via PeTTa's standard interop
!(import! &self random)

;; Define default mutation rate
(= (hyper-mutation-rate) 0.3)

;; =========================================
;; Scoring and Probability Utilities
;; =========================================

;; A node has STV (Strength, Confidence) in the Space 
;; (STV "A" 0.95 0.95)
(= (get-composite-score $feature)
   (match &self (STV $feature $s $c) 
          (/ (+ $s $c) 2.0)))

;; Fallback probabilistic score
(= (score-or-default $feature $default)
   (let $score (get-composite-score $feature)
        (if (== $score (empty)) $default $score)))

(= (prob-keep $feature)
   (let $score (score-or-default $feature -1.0)
        (if (> $score 0.0) $score (hyper-mutation-rate))))

(= (prob-flip $feature)
   (let $score (score-or-default $feature -1.0)
        (if (> $score 0.0) 
            (* (hyper-mutation-rate) (- 1.1 $score)) 
            (hyper-mutation-rate))))

;; =========================================
;; Mutation Operations
;; =========================================

;; 1. Additive Mutation (Flips elements to NOT)
;; (mutate-additive (AND A B))
(= (flip-feature $feat)
   (if (< (random.random) (prob-flip $feat))
       (NOT $feat)
       $feat))

;; Recurse through expression tuple
(= (mutate-add Nil) Nil)
(= (mutate-add (Cons $x $xs))
   (Cons (flip-feature $x) (mutate-add $xs)))

;; Entry point for an expression
(= (mutate-additive (AND $expr))
   (AND (mutate-add $expr)))
(= (mutate-additive (OR $expr))
   (OR (mutate-add $expr)))

;; 2. Multiplicative Mutation (Prunes elements)
(= (keep-feature? $feat)
   (< (random.random) (prob-keep $feat)))

(= (mutate-mul Nil) Nil)
(= (mutate-mul (Cons $x $xs))
   (if (keep-feature? $x)
       (Cons $x (mutate-mul $xs))
       (mutate-mul $xs)))

(= (mutate-multiplicative (AND $expr))
   (AND (mutate-mul $expr)))
(= (mutate-multiplicative (OR $expr))
   (OR (mutate-mul $expr)))


;; =========================================
;; Crossover Operation
;; =========================================

;; Set unions and masking in MeTTa via lists.
;; $m1 and $m2 are lists of features to cross over.

(= (element-in-list $x Nil) False)
(= (element-in-list $x (Cons $y $ys))
   (if (== $x $y) True (element-in-list $x $ys)))

;; Build unique Universe from two lists
(= (union Nil $set) $set)
(= (union (Cons $x $xs) $set)
   (if (element-in-list $x $set)
       (union $xs $set)
       (Cons $x (union $xs $set))))

;; Traverse universe and decide masking based on P and P_comp
(= (crossover-mask Nil $m1 $m2) Nil)
(= (crossover-mask (Cons $feat $universe) $m1 $m2)
   (let $p-keep (< (random.random) (prob-keep $feat))
        (let $in-m1 (element-in-list $feat $m1)
             (let $in-m2 (element-in-list $feat $m2)
                  (if $p-keep
                      ;; Keep from M1
                      (if $in-m1
                          (Cons $feat (crossover-mask $universe $m1 $m2))
                          (crossover-mask $universe $m1 $m2))
                      ;; Take from M2 (~P)
                      (if $in-m2
                          (Cons $feat (crossover-mask $universe $m1 $m2))
                          (crossover-mask $universe $m1 $m2)))))))

(= (crossover (AND $m1) (AND $m2))
   (let $universe (union $m1 $m2)
        (AND (crossover-mask $universe $m1 $m2))))
