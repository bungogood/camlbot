open Core
open Torch

let () =
  let x = Tensor.randn [ 4; 2 ] in
  let shape_text = Tensor.shape x |> List.map ~f:Int.to_string |> String.concat ~sep:"," in
  printf "torch ok, shape=[%s]\n%!" shape_text
