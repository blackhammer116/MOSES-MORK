;; =========================================
;; Fitness & Evaluator for MOSES-MORK
;; Native MeTTa Logic for Truth Tables
;; =========================================

;; -----------------------------------------
;; Boolean operations over Lists
;; Allows evaluation of S-expressions across all rows of the dataset
;; -----------------------------------------

(= (and-lists () ()) ())
(= (and-lists (:: $x $xs) (:: $y $ys))
   (:: (and $x $y) (and-lists $xs $ys)))

(= (or-lists () ()) ())
(= (or-lists (:: $x $xs) (:: $y $ys))
   (:: (or $x $y) (or-lists $xs $ys)))

(= (not-lists ()) ())
(= (not-lists (:: $x $xs))
   (:: (not $x) (not-lists $xs)))

;; -----------------------------------------
;; N-ary Evaluator
;; -----------------------------------------
;; Handles variable arity for AND / OR

(= (eval-and ()) ())
(= (eval-and (:: $arg ())) (evaluate $arg))
(= (eval-and (:: $arg $rest))
   (and-lists (evaluate $arg) (eval-and $rest)))

(= (eval-or ()) ())
(= (eval-or (:: $arg ())) (evaluate $arg))
(= (eval-or (:: $arg $rest))
   (or-lists (evaluate $arg) (eval-or $rest)))

;; -----------------------------------------
;; Expression Evaluator (AST traversal)
;; -----------------------------------------

;; Evaluate a raw symbol (fetches from TruthTable)
(= (evaluate $x)
   (if (== (get-type $x) Symbol)
       (match &self (TruthValue $x $list) $list)
       (empty)))

;; True/False constants
(= (evaluate True)
   (match &self (TruthValue Target $list) 
          ;; Hack to get an all-true list of matching length
          (not-lists (and-lists $list (not-lists $list)))
   ))
(= (evaluate False)
   (match &self (TruthValue Target $list) 
          (and-lists $list (not-lists $list))
   ))

;; Logical evaluators
(= (evaluate (NOT $x)) (not-lists (evaluate $x)))

;; N-ary dispatch (converting arguments using car/cdr recursive macros, or just predefined arities)
(= (evaluate (AND $x $y)) (and-lists (evaluate $x) (evaluate $y)))
(= (evaluate (AND $x $y $z)) (and-lists (evaluate $x) (and-lists (evaluate $y) (evaluate $z))))
(= (evaluate (AND $a $b $c $d)) (and-lists (evaluate $a) (evaluate (AND $b $c $d))))

(= (evaluate (OR $x $y)) (or-lists (evaluate $x) (evaluate $y)))
(= (evaluate (OR $x $y $z)) (or-lists (evaluate $x) (or-lists (evaluate $y) (evaluate $z))))
(= (evaluate (OR $a $b $c $d)) (or-lists (evaluate $a) (evaluate (OR $b $c $d))))

;; -----------------------------------------
;; Fitness Metrics
;; -----------------------------------------
;; Calculates how many elements match the target truth table

(= (count-matches () () $acc) $acc)
(= (count-matches (:: $x $xs) (:: $y $ys) $acc)
   (if (== $x $y)
       (count-matches $xs $ys (+ $acc 1))
       (count-matches $xs $ys $acc)))

(= (length () $acc) $acc)
(= (length (:: $x $xs) $acc)
   (length $xs (+ $acc 1)))

;; Computes fitness (matches / total_rows)
(= (fitness $expr)
   (let* (($pred (evaluate $expr))
          ($targ (match &self (TruthValue Target $list) $list))
          ($matches (count-matches $pred $targ 0))
          ($total (length $targ 0)))
     (/ $matches $total)))

;; Returns fully constructed Instance with score updated
(= (score-instance (Instance $expr $id $oldscore))
   (Instance $expr $id (fitness $expr)))
