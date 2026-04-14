open Core

let infer_representation input_dim =
  match input_dim with
  | 326 -> `Expanded
  | 198 -> `Modified
  | 196 -> `Original
  | _ -> failwithf "unsupported input dim %d" input_dim ()

let suffix_number ~prefix s =
  if String.equal s prefix then
    Some 0
  else
    match String.rsplit2 s ~on:'_' with
    | Some (p, n) when String.equal p prefix -> Int.of_string_opt n
    | _ -> None

let sort_connected ks =
  List.sort ks ~compare:(fun a b ->
    Int.compare
      (Option.value (suffix_number ~prefix:"connected" a) ~default:Int.max_value)
      (Option.value (suffix_number ~prefix:"connected" b) ~default:Int.max_value))

let () =
  let open Command.Let_syntax in
  Command.basic
    ~summary:"Convert legacy TF checkpoint exported as NPZ to Torch .ot"
    [%map_open
      let npz = flag "-npz" (required string) ~doc:"PATH input npz (from tf_ckpt_to_npz.py)"
      and out = flag "-out" (required string) ~doc:"PATH output torch checkpoint (.ot)"
      and activation = flag "-activation" (required string) ~doc:"Relu|Sigmoid hidden activation"
      in
      fun () ->
        let kernels =
          let in_file = Npy.Npz.open_in npz in
          Exn.protect
            ~f:(fun () ->
              Npy.Npz.entries in_file
              |> List.filter ~f:(String.is_prefix ~prefix:"connected_")
              |> sort_connected)
            ~finally:(fun () -> Npy.Npz.close_in in_file)
        in
        if List.is_empty kernels then failwith "no connected_* tensors found in npz";
        let input_dim =
          let first = Npy.Npz.open_in npz in
          Exn.protect
            ~f:(fun () ->
              let first_kernel = Npy.Npz.read first (List.hd_exn kernels) in
              let open Bigarray in
              match Npy.to_bigarray c_layout float32 first_kernel with
              | Some ba ->
                begin
                  match Genarray.dims ba with
                  | [| dim_in; _ |] -> dim_in
                  | _ -> failwith "expected rank-2 kernel"
                end
              | None ->
                begin
                  match Npy.to_bigarray c_layout float64 first_kernel with
                  | Some ba ->
                    begin
                      match Genarray.dims ba with
                      | [| dim_in; _ |] -> dim_in
                      | _ -> failwith "expected rank-2 kernel"
                    end
                  | None -> failwith "unsupported kernel dtype"
                end)
            ~finally:(fun () -> Npy.Npz.close_in first)
        in
        let hidden_layer_sizes =
          let in_file = Npy.Npz.open_in npz in
          Exn.protect
            ~f:(fun () ->
              let sizes =
                kernels
                |> List.map ~f:(fun key ->
                  let packed = Npy.Npz.read in_file key in
                  let open Bigarray in
                  match Npy.to_bigarray c_layout float32 packed with
                  | Some ba ->
                    begin
                      match Genarray.dims ba with
                      | [| _; dim_out |] -> dim_out
                      | _ -> failwith "expected rank-2 kernel"
                    end
                  | None ->
                    begin
                      match Npy.to_bigarray c_layout float64 packed with
                      | Some ba ->
                        begin
                          match Genarray.dims ba with
                          | [| _; dim_out |] -> dim_out
                          | _ -> failwith "expected rank-2 kernel"
                        end
                      | None -> failwith "unsupported kernel dtype"
                    end)
              in
              List.drop_last_exn sizes)
            ~finally:(fun () -> Npy.Npz.close_in in_file)
        in
        let representation = infer_representation input_dim in
        let activation =
          match String.lowercase activation with
          | "relu" -> `Relu
          | "sigmoid" -> `Sigmoid
          | _ -> failwith "activation must be Relu or Sigmoid"
        in
        let td = Td.create ~hidden_layer_sizes ~activation ~representation () in
        Td.load_legacy_npz td ~filename:npz;
        Td.save td ~filename:out;
        printf "converted %s -> %s\n%!" npz out
    ]
  |> Command_unix.run
