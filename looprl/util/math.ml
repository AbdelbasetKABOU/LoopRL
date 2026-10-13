open Base

(* Always return a nonnegative number. *)
(* Only return 0 if both are zero. *)
let rec gcd a b =
  if a = Int.min_value || b = Int.min_value then assert false
  else if a < 0 then gcd (-a) b
  else if b < 0 then gcd a (-b)
  else if a = 0 then b
  else if b = 0 then a
  else if a > b then gcd b (a % b)
  else gcd a (b % a)

let lcm a b =
  let d = gcd a b in
  if d = 0 then 0 else (a * b) / d