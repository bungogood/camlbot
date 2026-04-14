open Core

let () =
  Random.self_init ();
  let td =
    Td.create
      ~hidden_layer_sizes:[ 128; 128 ]
      ~activation:`Relu
      ~representation:`Modified
      ()
  in
  let replay = Replay_memory.create ~capacity:(Some 20_000) in
  for _ = 1 to 2_000 do
    let player = if Random.bool () then Player.Forwards else Player.Backwards in
    let to_play = if Random.bool () then Player.Forwards else Player.Backwards in
    let setup =
      { Equity.Setup.player
      ; to_play
      ; board = Board.starting
      }
    in
    let features = Td.Setup.create setup (Td.representation td) in
    let target = Random.float 1.0 in
    Replay_memory.enqueue replay (features, target)
  done;
  Td.train td replay ~minibatch_size:64 ~minibatches_number:50;
  let out = "models/td_torch_bootstrap.ot" in
  Td.save td ~filename:out;
  let td_loaded =
    Td.create
      ~hidden_layer_sizes:[ 128; 128 ]
      ~activation:`Relu
      ~representation:`Modified
      ()
  in
  Td.load td_loaded ~filename:out;
  let score =
    Td.eval td_loaded
      [| { Equity.Setup.player = Player.Forwards
         ; to_play = Player.Forwards
         ; board = Board.starting
         }
      |]
    |> fun xs -> xs.(0)
  in
  printf "saved torch checkpoint: %s\nloaded score=%0.4f\n%!" out score
