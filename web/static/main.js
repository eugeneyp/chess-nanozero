/**
 * Chess NanoZero frontend — game logic and API integration.
 *
 * Architecture:
 * - chess.js (game) is the authoritative state machine for move validation,
 *   game-over detection, and FEN serialisation.
 * - chessboard.js (board) is purely visual; it is synced to game.fen() after
 *   every move.
 * - The backend is called with the current FEN; it is stateless per request.
 *
 * Player is always White; AlphaZero engine is always Black.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/** chess.js game state — source of truth for rules and FEN */
let game = new Chess();

/** chessboard.js board instance — visual only */
let board = null;

/** Blocks interaction while the engine is searching */
let isThinking = false;

/** Currently selected square (click-to-move state machine, mobile only) */
let selectedSquare = null;

/**
 * True on touch/mobile devices (coarse pointer), false on desktop (fine pointer/mouse).
 * Determines whether to use click-to-move or drag-and-drop.
 */
const isMobile = window.matchMedia('(pointer: coarse)').matches;

// ---------------------------------------------------------------------------
// chessboard.js callbacks
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Drag-and-drop callbacks (desktop only)
// ---------------------------------------------------------------------------

function onDragStart(source, piece) {
  if (isThinking) return false;
  if (game.game_over()) return false;
  if (piece.startsWith('b')) return false;
  return true;
}

function onDrop(source, target) {
  const move = game.move({ from: source, to: target, promotion: 'q' });
  if (move === null) return 'snapback';
  updateStatus();
  if (!game.game_over()) requestEngineMove();
}

function onSnapEnd() {
  board.position(game.fen());
}

// ---------------------------------------------------------------------------
// Click-to-move callbacks (mobile only)
// ---------------------------------------------------------------------------

/**
 * Core click-to-move state machine.
 * Called by the #board click handler with the algebraic square and piece string.
 *
 * @param {string} square - The clicked square (e.g. "e2")
 * @param {string|null} piece - chessboard.js piece string (e.g. "wP") or null
 */
function onSquareClick(square, piece) {
  if (isThinking || game.game_over()) return;

  if (selectedSquare === null) {
    // Phase 1: select own piece
    if (!piece || piece.startsWith('b')) return;
    selectedSquare = square;
    highlightSquare(square, 'selected');
    game.moves({ square: square, verbose: true })
        .forEach(function(m) { highlightSquare(m.to, 'legal'); });
  } else {
    // Phase 2: destination click
    const legalDests = game.moves({ square: selectedSquare, verbose: true })
                           .map(function(m) { return m.to; });
    if (legalDests.includes(square)) {
      // Valid destination → make move (auto-promote to queen)
      game.move({ from: selectedSquare, to: square, promotion: 'q' });
      clearHighlights();
      selectedSquare = null;
      board.position(game.fen());
      updateStatus();
      if (!game.game_over()) requestEngineMove();
    } else if (piece && piece.startsWith('w')) {
      // Clicked another own piece → switch selection
      clearHighlights();
      selectedSquare = square;
      highlightSquare(square, 'selected');
      game.moves({ square: square, verbose: true })
          .forEach(function(m) { highlightSquare(m.to, 'legal'); });
    } else {
      // Clicked empty/opponent square → cancel
      clearHighlights();
      selectedSquare = null;
    }
  }
}

/**
 * Convert a chess.js piece object to a chessboard.js piece string.
 * chess.js: { type: 'p', color: 'w' } → chessboard.js: 'wP'
 */
function pieceString(p) {
  if (!p) return null;
  return p.color + p.type.toUpperCase();
}

// ---------------------------------------------------------------------------
// Highlight helpers
// ---------------------------------------------------------------------------

function highlightSquare(square, type) {
  const el = document.querySelector('[data-square="' + square + '"]');
  if (el) el.classList.add('highlight-' + type);
}

function clearHighlights() {
  document.querySelectorAll('.highlight-selected, .highlight-legal')
    .forEach(function(el) { el.classList.remove('highlight-selected', 'highlight-legal'); });
}

// ---------------------------------------------------------------------------
// Engine communication
// ---------------------------------------------------------------------------

/**
 * Fetch the AlphaZero engine's best move for the current position via POST /api/move.
 *
 * Updates the board and simulations display once the response arrives.
 * Sets isThinking to block user interaction during the search.
 */
function requestEngineMove() {
  isThinking = true;
  setStatus('AlphaZero is thinking...');

  const timeLimit = parseFloat(document.getElementById('time-select').value);

  fetch('/api/move', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen: game.fen(), time_limit: timeLimit }),
  })
    .then(function (response) {
      if (!response.ok) {
        return response.json().then(function (err) {
          throw new Error(err.detail || 'Server error');
        });
      }
      return response.json();
    })
    .then(function (data) {
      // Parse the UCI move string (e.g. "e7e5" or "e7e8q")
      const from = data.move.slice(0, 2);
      const to   = data.move.slice(2, 4);
      const promo = data.move.length > 4 ? data.move[4] : 'q';

      game.move({ from: from, to: to, promotion: promo });
      board.position(game.fen());

      // Update simulations count
      if (data.simulations !== undefined) {
        document.getElementById('sims-value').textContent = data.simulations;
      }

      isThinking = false;
      updateStatus();
    })
    .catch(function (err) {
      console.error('Engine request failed:', err);
      setStatus('Error: ' + err.message);
      isThinking = false;
    });
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

/**
 * Update the status display based on the current game state.
 * Checks for checkmate, stalemate, draw conditions, and whose turn it is.
 */
function updateStatus() {
  if (game.in_checkmate()) {
    // If it's Black's turn and checkmate, White (player) won
    setStatus(game.turn() === 'b' ? 'You win! Checkmate.' : 'AlphaZero wins! Checkmate.');
    return;
  }

  if (game.in_stalemate()) {
    setStatus('Draw — stalemate.');
    return;
  }

  if (game.in_threefold_repetition()) {
    setStatus('Draw — threefold repetition.');
    return;
  }

  if (game.insufficient_material()) {
    setStatus('Draw — insufficient material.');
    return;
  }

  if (game.in_draw()) {
    setStatus('Draw.');
    return;
  }

  if (game.turn() === 'w') {
    const suffix = game.in_check() ? ' (in check)' : '';
    setStatus('Your turn' + suffix);
  } else {
    setStatus('AlphaZero is thinking...');
  }
}

/**
 * Set the status text in the status panel.
 * @param {string} text - Status message to display
 */
function setStatus(text) {
  document.getElementById('status-text').textContent = text;
}

/**
 * Undo the last two half-moves (engine's reply + player's move).
 *
 * Takes back one full exchange so it's the player's turn again.
 * No-ops if the engine is thinking or fewer than 2 moves have been played.
 */
function takeBack() {
  if (isThinking) return;
  if (game.history().length < 2) return;

  game.undo(); // removes engine's last move
  game.undo(); // removes player's last move
  if (isMobile) { clearHighlights(); selectedSquare = null; }
  board.position(game.fen());
  updateStatus();
}

/**
 * Reset the game to the starting position and clear all UI state.
 */
function newGame() {
  game.reset();
  board.start();
  if (isMobile) { clearHighlights(); selectedSquare = null; }
  isThinking = false;
  document.getElementById('sims-value').textContent = '—';
  setStatus('Your turn');
}

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/**
 * Initialise chessboard.js once the DOM is ready.
 * Desktop: drag-and-drop. Mobile (coarse pointer): click-to-move.
 */
$(document).ready(function () {
  const config = isMobile
    ? {
        draggable: false,
        position: 'start',
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
      }
    : {
        draggable: true,
        position: 'start',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
      };

  board = Chessboard('board', config);

  // Click-to-move: event delegation on the board container (mobile only).
  // chessboard.js does not support onSquareClick, so we listen on #board
  // and find the clicked square via its data-square attribute.
  if (isMobile) {
    document.getElementById('board').addEventListener('click', function(e) {
      const squareEl = e.target.closest('[data-square]');
      if (!squareEl) return;
      const square = squareEl.dataset.square;
      const p = game.get(square);
      onSquareClick(square, pieceString(p));
    });
  }

  // Wire up buttons
  document.getElementById('take-back-btn').addEventListener('click', takeBack);
  document.getElementById('new-game-btn').addEventListener('click', newGame);
});
