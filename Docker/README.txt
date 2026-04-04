peak

{
  "distances": [              <-- 正面から左右2度ずつ、計7本のrayのdistance
    1658.11767578125,
    1605.6962890625,
    1548.2283935546875,
    1149.097900390625,
    1449.81103515625,
    1407.6173095703125,
    1369.43115234375
  ],
  --- "distToWall": -1166.2440185546875,              <--
  wallDistL
  wallDistR
  
  "isTargetVisible": true,                          <-- 敵が見えているときはtrueのフラグ
  "timeTargetVisible": 0.017174899578094482,         <-- 敵が見えてからの時間
  "targetDistance": 1168.5311970454354,           <-- 敵までの距離
  "targetVelocity": {              <-- 敵の velocity
    "x": 0,
    "y": 0,
    "z": 0
  },
  "currentAimError": {                <-- aim のずれ
    "x": -0.11505126953125,
    "y": -3.5896973609924316,
    "z": 0
  },
  "myVelocity": {               <-- 自分の velocity
    "x": 0,
    "y": 0,
    "z": 0
  },
  "myPitch": 9.520000457763672,    <-- 自分の pitch
   
  --"isReloading": false,            <-- リロード中フラグ
  "moveRight": 0,                  <-- 正解用データ  左右の移動
  "moveForward": 1,                <-- 正解用データ 前後の移動
  
  
  -- "isStoppingTrigger": true,       <-- 正解用データ 止まっているときは true
  "isFire": false,                 <-- 正解用データ 射撃しているときは true
  "isCrouching": false,            <-- 正解用データ しゃがんでいるときは true

  "myTurn": -0.14000000059604645,  <-- 自分のマウス turn
  "myLockup": -0.14000000059604645   <-- 自分のマウス lookup
}