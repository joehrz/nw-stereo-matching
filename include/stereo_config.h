#pragma once
/*  Middlebury‑2005 “cone” constants
    Positive disparity  =  x_left  –  x_right
    All host / device code must use these symbols.                           */
constexpr float DISP_SCALE = 1.0f / 16.0f;   // ground‑truth stored as d*16
constexpr int   INVALID_GT = 0;              // 0  means  “unknown disparity”