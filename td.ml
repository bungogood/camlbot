open Core
open Torch

type t =
  { representation : [ `Original | `Modified | `Expanded ]
  ; vs : Var_store.t
  ; model : Layer.t
  ; optimizer : Optimizer.t
  }

let default_device () =
  match Sys.getenv "CAMLBOT_DEVICE" with
  | None -> Device.Cpu
  | Some s -> Device.of_string s

let create ?device ?(epsilon_init = 0.1) ~hidden_layer_sizes ~activation ~representation () =
  let input_size =
    match representation with
    | `Original -> 196
    | `Modified -> 198
    | `Expanded -> 326
  in
  let hidden_activation =
    match activation with
    | `Sigmoid -> Layer.Sigmoid
    | `Relu -> Layer.Relu
  in
  let device = Option.value device ~default:(default_device ()) in
  let vs = Var_store.create ~device ~name:"td" () in
  let layer_sizes = input_size :: hidden_layer_sizes @ [ 1 ] in
  let layer_size_pairs =
    List.zip_exn
      (List.take layer_sizes (List.length layer_sizes - 1))
      (List.tl_exn layer_sizes)
  in
  let layers =
    List.mapi layer_size_pairs ~f:(fun i (dim_in, dim_out) ->
      let activation =
        if Int.equal i (List.length hidden_layer_sizes) then
          Some Layer.Sigmoid
        else
          Some hidden_activation
      in
      Layer.linear
        vs
        ?activation
        ~w_init:(Normal { mean = 0.; stdev = epsilon_init })
        ~input_dim:dim_in
        dim_out)
  in
  let model = Layer.sequential layers in
  let optimizer = Optimizer.adam vs ~learning_rate:0.001 in
  { representation; vs; model; optimizer }

let representation t = t.representation

module Setup = struct
  type t =
    { board : float array
    ; sign : float
    }
  [@@deriving sexp]

  let create { Equity.Setup.player; to_play; board } version =
    { board = Array.of_list (Board.to_representation board version ~to_play)
    ; sign = if Player.equal to_play player then 1. else -1.
    }

  let modifier ~sign valuation = Float.(+) 0.5 (Float.( * ) Float.(valuation - 0.5) sign)

  module And_valuation = struct
    type nonrec t = t * float [@@deriving sexp]
  end
end

let eval t equity_setups =
  let boards, signs =
    Array.map equity_setups ~f:(fun equity_setup ->
      let { Setup.board; sign } = Setup.create equity_setup t.representation in
      board, sign)
    |> Array.unzip
  in
  let device = Var_store.device t.vs in
  let outputs =
    Tensor.no_grad (fun () ->
      boards
      |> Tensor.of_float2
      |> Tensor.to_device ~device
      |> Layer.forward t.model
      |> Tensor.to_device ~device:Device.Cpu
      |> Tensor.to_float2_exn)
  in
  Array.map2_exn signs outputs ~f:(fun sign valuation -> Setup.modifier ~sign valuation.(0))

let train t replay_memory ~minibatch_size ~minibatches_number =
  let device = Var_store.device t.vs in
  for _ = 1 to minibatches_number do
    let (boards, signs), valuations =
      Replay_memory.sample replay_memory minibatch_size
      |> List.map ~f:(fun ({ Setup.board; sign }, valuation) -> (board, sign), valuation)
      |> Array.of_list
      |> Array.unzip
      |> Tuple2.map_fst ~f:Array.unzip
    in
    let modified_valuations =
      Array.map2_exn signs valuations ~f:(fun sign valuation -> [| Setup.modifier ~sign valuation |])
    in
    let xs = Tensor.of_float2 boards |> Tensor.to_device ~device in
    let ys = Tensor.of_float2 modified_valuations |> Tensor.to_device ~device in
    let preds = Layer.forward t.model xs in
    let loss = Tensor.mse_loss preds ys in
    Optimizer.backward_step t.optimizer ~loss
  done

let save t ~filename =
  let named_tensors = Var_store.all_vars t.vs in
  Serialize.save_multi ~named_tensors ~filename

let load t ~filename =
  let named_tensors = Var_store.all_vars t.vs in
  Serialize.load_multi_ ~named_tensors ~filename

let load_legacy_npz t ~filename =
  let parse_layer_index ~base s =
    if String.equal s base then
      Some 0
    else
      match String.rsplit2 s ~on:'_' with
      | Some (prefix, suffix) when String.equal prefix base -> Int.of_string_opt suffix
      | _ -> None
  in
  let sort_by_suffix ?base pairs =
    List.sort pairs ~compare:(fun (a, _) (b, _) ->
      Int.compare
        (Option.value
           (match base with
            | None ->
              begin
                match String.rsplit2 a ~on:'_' with
                | Some (_, suffix) -> Int.of_string_opt suffix
                | None -> None
              end
            | Some base -> parse_layer_index ~base a)
           ~default:Int.max_value)
        (Option.value
           (match base with
            | None ->
              begin
                match String.rsplit2 b ~on:'_' with
                | Some (_, suffix) -> Int.of_string_opt suffix
                | None -> None
              end
            | Some base -> parse_layer_index ~base b)
           ~default:Int.max_value))
  in
  let to_float_array1 packed =
    let open Bigarray in
    match Npy.to_bigarray c_layout float32 packed with
    | Some ga ->
      begin
        match Genarray.dims ga with
        | [| n |] -> Array.init n ~f:(fun i -> Genarray.get ga [| i |])
        | [| 1; n |] -> Array.init n ~f:(fun i -> Genarray.get ga [| 0; i |])
        | [| n; 1 |] -> Array.init n ~f:(fun i -> Genarray.get ga [| i; 0 |])
        | _ -> failwith "expected rank-1 tensor (or degenerate rank-2)"
      end
    | None ->
      begin
        match Npy.to_bigarray c_layout float64 packed with
        | Some ga ->
          begin
            match Genarray.dims ga with
            | [| n |] -> Array.init n ~f:(fun i -> Genarray.get ga [| i |])
            | [| 1; n |] -> Array.init n ~f:(fun i -> Genarray.get ga [| 0; i |])
            | [| n; 1 |] -> Array.init n ~f:(fun i -> Genarray.get ga [| i; 0 |])
            | _ -> failwith "expected rank-1 tensor (or degenerate rank-2)"
          end
        | None -> failwith "unsupported npz dtype for 1d tensor"
      end
  in
  let to_float_array2_transposed packed =
    let open Bigarray in
    match Npy.to_bigarray c_layout float32 packed with
    | Some ga ->
      begin
        match Genarray.dims ga with
        | [| dim_in; dim_out |] ->
          Array.init dim_out ~f:(fun o -> Array.init dim_in ~f:(fun i -> Genarray.get ga [| i; o |]))
        | _ -> failwith "expected rank-2 tensor"
      end
    | None ->
      begin
        match Npy.to_bigarray c_layout float64 packed with
        | Some ga ->
          begin
            match Genarray.dims ga with
            | [| dim_in; dim_out |] ->
              Array.init dim_out ~f:(fun o -> Array.init dim_in ~f:(fun i -> Genarray.get ga [| i; o |]))
            | _ -> failwith "expected rank-2 tensor"
          end
        | None -> failwith "unsupported npz dtype for 2d tensor"
      end
  in
  let in_file = Npy.Npz.open_in filename in
  Exn.protect
    ~f:(fun () ->
      let entries = Npy.Npz.entries in_file in
      let kernels =
        entries
        |> List.filter ~f:(String.is_prefix ~prefix:"connected_")
        |> List.map ~f:(fun key -> key, Npy.Npz.read in_file key)
        |> sort_by_suffix
      in
      let biases =
        entries
        |> List.filter ~f:(String.is_prefix ~prefix:"bias_")
        |> List.map ~f:(fun key -> key, Npy.Npz.read in_file key)
        |> sort_by_suffix
      in
      let vars = Var_store.all_vars t.vs in
      let weight_vars =
        vars
        |> List.filter ~f:(fun (_, tensor) -> Int.equal (List.length (Tensor.shape tensor)) 2)
        |> sort_by_suffix ~base:"weight"
      in
      let bias_vars =
        vars
        |> List.filter ~f:(fun (_, tensor) -> Int.equal (List.length (Tensor.shape tensor)) 1)
        |> sort_by_suffix ~base:"bias"
      in
      if not (Int.equal (List.length kernels) (List.length weight_vars)) then
        failwith "legacy checkpoint kernel count mismatch";
      if not (Int.equal (List.length biases) (List.length bias_vars)) then
        failwith "legacy checkpoint bias count mismatch";
      Tensor.no_grad (fun () ->
        List.iter2_exn weight_vars kernels ~f:(fun (dst_name, dst) (src_name, src_packed) ->
          let src = Tensor.of_float2 (to_float_array2_transposed src_packed) in
          if not (List.equal Int.equal (Tensor.shape dst) (Tensor.shape src)) then
            failwithf "shape mismatch for %s <- %s: dst=%s src=%s"
              dst_name src_name (Tensor.shape_str dst) (Tensor.shape_str src) ();
          Tensor.copy_ dst ~src);
        List.iter2_exn bias_vars biases ~f:(fun (dst_name, dst) (src_name, src_packed) ->
          let src = Tensor.of_float1 (to_float_array1 src_packed) in
          if not (List.equal Int.equal (Tensor.shape dst) (Tensor.shape src)) then
            failwithf "shape mismatch for %s <- %s: dst=%s src=%s"
              dst_name src_name (Tensor.shape_str dst) (Tensor.shape_str src) ();
          Tensor.copy_ dst ~src)))
    ~finally:(fun () -> Npy.Npz.close_in in_file)

let var_shapes t =
  Var_store.all_vars t.vs |> List.map ~f:(fun (name, tensor) -> name, Tensor.shape tensor)

let sexp_of_vars t =
  Var_store.all_vars t.vs
  |> List.map ~f:(fun (name, tensor) -> name, Tensor.to_string tensor ~line_size:120)
  |> [%sexp_of:(string * string) list]
