[function]
& : 2
| : 2
~ : 1
simp : 2
sconj : 2

[variable]
x y z u v w

[axiom]
(commM) &(x y) = &(y x)
(commJ) |(x y) = |(y x)
(assocM) &(x &(y z)) = &(&(x y) z)
(assocJ) |(x |(y z)) = |(|(x y) z)
(absorbM) &(x |(x y)) = x
(absorbJ) |(x &(x y)) = x
(negneg) ~(~(x)) = x
(deMorgenM) ~(&(x y)) = |(~(x) ~(y))
(deMorgenJ) ~(|(x y)) = &(~(x) ~(y))
(OML) |(x y) = |(&(|(x y) x) &(|(x y) ~(x)))
(SIMP) simp(x y) = |(~(x) &(x y))
(SCONJ) sconj(x y) = &(x |(~(x) y))