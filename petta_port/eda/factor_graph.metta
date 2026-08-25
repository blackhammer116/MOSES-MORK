;; =========================================
;; Factor Graph & Dependency Miner
;; Native Belief Propagation in MeTTa
;; =========================================

;; Python math functions via PeTTa interop
!(import! &self math)

;; -----------------------------------------
;; Beta Distribution State Utilities
;; -----------------------------------------
;; Nodes are stored iteratively as: (NodeState "A" 1.0 1.0)
;; Meaning (NodeState <name> <alpha> <beta>)

(= (get-strength (NodeState $n $a $b)) 
   (/ $a (+ $a $b)))

(= (get-confidence (NodeState $n $a $b))
   (let $total (+ $a $b)
        (/ $total (+ $total 1.0))))

;; Helper to wrap evidence counts back to probabilities natively
(= (evidence-to-beta $stv-strength $stv-confidence $base-counts)
   (let* (($evidence (max 0.1 (* $stv-confidence $base-counts)))
          ($a (+ (* $stv-strength $evidence) 1.0))
          ($b (+ (* (- 1.0 $stv-strength) $evidence) 1.0)))
         (Cons $a $b)))

;; -----------------------------------------
;; Factor Dependencies (Modus Ponens/Tollens)
;; -----------------------------------------
;; Rules are atoms in Space: (Rule "A" "B" <strength> <conf>)
  
;; Forward Message Pass (Src -> Dst)
(= (forward-msg $src-state $rule-s $rule-c)
   (let* (($p-src (get-strength $src-state))
          ($src-ev (+ (match $src-state (NodeState $n $a $b) $a) 
                      (match $src-state (NodeState $n $a $b) $b)))
          ($rule-cap (* $rule-c 20.0))
          ($fwd-s (+ (* $p-src $rule-s) (* (- 1.0 $p-src) 0.5)))
          ($fwd-ev (min (* $src-ev 0.9) $rule-cap)))
         (Cons (* $fwd-s $fwd-ev) (* (- 1.0 $fwd-s) $fwd-ev))))

;; Backward Message Pass (Dst -> Src - Abduction)
(= (backward-msg $dst-state $rule-s $rule-c)
   (let* (($p-dst (get-strength $dst-state))
          ($dst-ev (+ (match $dst-state (NodeState $n $a $b) $a) 
                      (match $dst-state (NodeState $n $a $b) $b)))
          ($rule-cap (* $rule-c 20.0))
          ($bwd-s (+ (* $p-dst $rule-s) (* (- 1.0 $p-dst) (- 1.0 $rule-s))))
          ($bwd-ev (min (* $dst-ev (* 0.9 0.5)) $rule-cap)))
         (Cons (* $bwd-s $bwd-ev) (* (- 1.0 $bwd-s) $bwd-ev))))

;; -----------------------------------------
;; Dependency Miner 
;; -----------------------------------------
;; Mathematical Estimations
(= (sigmoid $x)
   (/ 1.0 (+ 1.0 (math.exp (* -1.0 $x)))))

(= (calc-pmi $count-xy $count-x $count-y $total)
   (let* (($p-x (/ $count-x $total))
          ($p-y (/ $count-y $total))
          ($p-xy (/ $count-xy $total))
          ($lift (/ $p-xy (* $p-x $p-y))))
         ;; Error prevention wrapper if $lift <= 0
         (if (<= $lift 0) 0.0 (math.log2 $lift))))
         
(= (mine-rule $sym1 $sym2 $count-xy $count-x $count-y $total)
   (let* (($pmi (calc-pmi $count-xy $count-x $count-y $total))
           ($strength (sigmoid $pmi))
           ($confidence (/ $count-xy $total)))
         (if (>= $pmi 0.1)
             (Rule $sym1 $sym2 $strength $confidence)
             (empty))))
