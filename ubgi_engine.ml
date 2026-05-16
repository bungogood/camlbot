open Core

let reply line =
  print_endline line;
  Out_channel.flush stdout

let starts_with ~prefix s = String.is_prefix s ~prefix

let parse_dice s =
  let parts =
    String.split (String.strip s) ~on:' '
    |> List.filter ~f:(fun x -> not (String.is_empty x))
  in
  match parts with
  | [d1; d2] ->
    begin
      match Int.of_string_opt d1, Int.of_string_opt d2 with
      | Some i, Some j when Int.between i ~low:1 ~high:6 && Int.between j ~low:1 ~high:6 ->
        if Int.equal i j then Some (Roll.Double i) else Some (Roll.High_low (Int.max i j, Int.min i j))
      | _ -> None
    end
  | _ -> None

let move_to_text move =
  let src_text, dst_text =
    match Move.from move with
    | `Bar ->
      let dst = Int.(25 - Move.uncapped_distance move) in
      "bar", Int.to_string dst
    | `Position src ->
      let dst = Int.(src - Move.uncapped_distance move) in
      let dst_text = if Int.(dst <= 0) then "off" else Int.to_string dst in
      Int.to_string src, dst_text
  in
  sprintf "%s/%s" src_text dst_text

type eval_mode =
  | Pipcount
  | Td of
      { hidden_layer_sizes : int list
      ; activation : [ `Sigmoid | `Relu ]
      ; representation : [ `Original | `Modified | `Expanded ]
      ; ckpt : string
      ; look_ahead : int
      ; device : Torch.Device.t option
      }

type runtime_eval =
  | Pipcount_eval of
      { mutable look_ahead : int
      ; mutable equity_fn : Equity.t
      }
  | Td_eval of
      { td : Td.t
      ; mutable look_ahead : int
      ; mutable equity_fn : Equity.t
      }

let parse_hidden_sizes s =
  String.split s ~on:','
  |> List.filter ~f:(fun x -> not (String.is_empty (String.strip x)))
  |> List.map ~f:String.strip
  |> List.map ~f:Int.of_string

let parse_activation s =
  match String.lowercase (String.strip s) with
  | "relu" -> `Relu
  | "sigmoid" -> `Sigmoid
  | _ -> failwith "--activation must be relu or sigmoid"

let parse_representation s =
  match String.lowercase (String.strip s) with
  | "expanded" -> `Expanded
  | "modified" -> `Modified
  | "original" -> `Original
  | _ -> failwith "--representation must be expanded|modified|original"

let parse_device s = Torch.Device.of_string (String.strip s)

let parse_eval_mode () =
  let argv = Sys.get_argv () in
  let arg_value i =
    if i + 1 >= Array.length argv then
      failwithf "missing value for %s" argv.(i) ()
    else
      argv.(i + 1)
  in
  let rec loop i hidden_sizes activation representation ckpt look_ahead device =
    if i >= Array.length argv then
      match ckpt with
      | None -> Pipcount
      | Some ckpt_value ->
        let hidden_layer_sizes =
          match hidden_sizes with
          | Some hs -> hs
          | None -> failwith "--hidden is required when --ckpt is provided"
        in
        let activation =
          match activation with
          | Some a -> a
          | None -> failwith "--activation is required when --ckpt is provided"
        in
        let representation =
          match representation with
          | Some r -> r
          | None -> failwith "--representation is required when --ckpt is provided"
        in
        Td
          { hidden_layer_sizes
          ; activation
          ; representation
          ; ckpt = ckpt_value
          ; look_ahead
          ; device
          }
    else
      match argv.(i) with
      | "--hidden" -> loop (i + 2) (Some (parse_hidden_sizes (arg_value i))) activation representation ckpt look_ahead device
      | "--activation" -> loop (i + 2) hidden_sizes (Some (parse_activation (arg_value i))) representation ckpt look_ahead device
      | "--representation" -> loop (i + 2) hidden_sizes activation (Some (parse_representation (arg_value i))) ckpt look_ahead device
      | "--ckpt" -> loop (i + 2) hidden_sizes activation representation (Some (arg_value i)) look_ahead device
      | "--look-ahead" -> loop (i + 2) hidden_sizes activation representation ckpt (Int.of_string (arg_value i)) device
      | "--device" -> loop (i + 2) hidden_sizes activation representation ckpt look_ahead (Some (parse_device (arg_value i)))
      | unknown -> failwithf "unknown arg: %s" unknown ()
  in
  loop 1 None None None None 1 None

let current_eval =
  match parse_eval_mode () with
  | Pipcount ->
    Pipcount_eval
      { look_ahead = 1
      ; equity_fn = Equity.minimax Equity.pip_count_ratio ~look_ahead:1 Outcome.Game
      }
  | Td { hidden_layer_sizes; activation; representation; ckpt; look_ahead; device } ->
    let td = Td.create ?device ~hidden_layer_sizes ~activation ~representation () in
    Td.load td ~filename:ckpt;
    Td_eval
      { td
       ; look_ahead
       ; equity_fn = Equity.minimax' (Td.eval td) ~look_ahead Outcome.Game
       }

let score_one ~player ~board =
  let setup =
    { Equity.Setup.player
    ; to_play = Player.flip player
    ; board
    }
  in
  match current_eval with
  | Pipcount_eval { equity_fn; _ } -> Equity.eval equity_fn setup
  | Td_eval { look_ahead; td; equity_fn } ->
    if Int.equal look_ahead 1 then
      Td.eval td [| setup |] |> fun xs -> xs.(0)
    else
      Equity.eval equity_fn setup

let score_many ~player boards =
  match current_eval with
  | Pipcount_eval { equity_fn; _ } ->
    List.map boards ~f:(fun board ->
      Equity.eval equity_fn
        { Equity.Setup.player
        ; to_play = Player.flip player
        ; board
        })
  | Td_eval { look_ahead; td; equity_fn } ->
    if Int.equal look_ahead 1 then
      boards
      |> List.map ~f:(fun board ->
           { Equity.Setup.player
           ; to_play = Player.flip player
           ; board
           })
      |> Array.of_list
      |> Td.eval td
      |> Array.to_list
    else
      List.map boards ~f:(fun board ->
        Equity.eval equity_fn
          { Equity.Setup.player
          ; to_play = Player.flip player
          ; board
          })

let choose_best_turn ~player ~board ~roll =
  let turns = Move.all_legal_turns roll player board in
  match turns with
  | [] -> `Pass
  | [([], _)] -> `Pass
  | _ ->
    let non_empty_turns = List.filter turns ~f:(fun (moves, _) -> not (List.is_empty moves)) in
    let boards = List.map non_empty_turns ~f:snd in
    let scores = score_many ~player boards in
    let scored_turns = List.map2_exn non_empty_turns scores ~f:(fun (moves, _) score -> moves, score) in
    let best_turn = List.max_elt scored_turns ~compare:(fun (_, s1) (_, s2) -> Float.compare s1 s2) in
    match best_turn with
    | None -> `Pass
    | Some (moves, _) ->
      moves
      |> List.map ~f:move_to_text
      |> String.concat ~sep:" "
      |> fun text -> `Move text

let set_ply ply =
  if ply < 1 || ply > 2 then
    Error "error bad_value engine.ply"
  else (
    (match current_eval with
     | Pipcount_eval r ->
       if not (Int.equal r.look_ahead ply) then begin
         r.look_ahead <- ply;
         r.equity_fn <- Equity.minimax Equity.pip_count_ratio ~look_ahead:ply Outcome.Game
       end
     | Td_eval r ->
       if not (Int.equal r.look_ahead ply) then begin
         r.look_ahead <- ply;
         r.equity_fn <- Equity.minimax' (Td.eval r.td) ~look_ahead:ply Outcome.Game
       end);
    Ok ())

let () =
  let board = ref None in
  let to_play = ref Player.Backwards in
  let roll = ref None in
  In_channel.iter_lines In_channel.stdin ~f:(fun raw ->
    let cmd = String.strip raw in
    if String.is_empty cmd then
      ()
    else if String.equal cmd "ubgi" then begin
      reply "id name camlbot 0.1";
      reply "id author jacobhilton";
      reply "proto 0.2";
      let default_ply =
        match current_eval with
        | Pipcount_eval { look_ahead; _ } -> look_ahead
        | Td_eval { look_ahead; _ } -> look_ahead
      in
      reply "key game.variant enum backgammon backgammon";
      reply (sprintf "key engine.ply int 1..2 %d" default_ply);
      reply "ubgiok"
    end else if String.equal cmd "keys" then begin
      reply "key game.variant enum backgammon backgammon";
      reply "key engine.ply int 1..2 1"
    end else if starts_with ~prefix:"get " cmd then begin
      let key = String.drop_prefix cmd 4 |> String.strip in
      if String.equal key "game.variant" then reply "value game.variant backgammon"
      else if String.equal key "engine.ply" then (
        let p =
          match current_eval with
          | Pipcount_eval { look_ahead; _ } -> look_ahead
          | Td_eval { look_ahead; _ } -> look_ahead
        in
        reply (sprintf "value engine.ply %d" p)
      )
      else reply "error unsupported key"
    end else if starts_with ~prefix:"set " cmd then begin
      let parts = String.split (String.strip cmd) ~on:' ' |> List.filter ~f:(fun x -> not (String.is_empty x)) in
      match parts with
      | _::key::value_parts ->
        let value = String.concat ~sep:" " value_parts in
        if String.equal key "game.variant" then begin
          if String.equal (String.lowercase (String.strip value)) "backgammon" then ()
          else reply "error bad_value game.variant"
        end else if String.equal key "engine.ply" then begin
          match Int.of_string_opt (String.strip value) with
          | None -> reply "error bad_value engine.ply"
          | Some p ->
            begin match set_ply p with
            | Ok () -> ()
            | Error _ -> reply "error bad_value engine.ply"
            end
        end else reply "error unsupported key"
      | _ -> reply "error bad_command set"
    end else if String.equal cmd "isready" then begin
      reply "readyok"
    end else if String.equal cmd "newgame" then begin
      board := None;
      roll := None
    end else if String.equal cmd "quit" then begin
      exit 0
    end else if starts_with ~prefix:"position gnubgid " cmd then begin
      let id = String.drop_prefix cmd (String.length "position gnubgid ") |> String.strip in
      begin
        match Board.of_gnubgid id with
        | Ok (new_board, `To_play player) ->
          board := Some new_board;
          to_play := player;
          roll := None
        | Error _ -> reply "error bad_value position"
      end
    end else if String.equal cmd "position xgid" || starts_with ~prefix:"position xgid " cmd then begin
      reply "error unsupported position.xgid"
    end else if starts_with ~prefix:"dice " cmd then begin
      let dice = String.drop_prefix cmd (String.length "dice ") in
      match parse_dice dice with
      | None ->
        roll := None;
        reply "error bad_value dice"
      | Some new_roll -> roll := Some new_roll
    end else if String.equal cmd "go" || starts_with ~prefix:"go " cmd then begin
      if String.is_substring cmd ~substring:"role cube"
         || String.is_substring cmd ~substring:"role turn"
      then
        reply "error unsupported role"
      else
        match !board, !roll with
        | None, _ -> reply "error bad_state missing.position"
        | _, None -> reply "error bad_state missing.dice"
        | Some b, Some r ->
          begin
            match choose_best_turn ~player:!to_play ~board:b ~roll:r with
            | `Pass -> reply "bestmove pass"
            | `Move mv -> reply ("bestmove " ^ mv)
          end
    end else begin
      reply "error bad_command unknown"
    end)
