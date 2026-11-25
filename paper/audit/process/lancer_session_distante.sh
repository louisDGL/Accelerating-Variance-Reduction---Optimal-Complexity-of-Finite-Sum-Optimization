#!/bin/bash
# Lance la session distante de l'audit dans tmux (session "audit"), independante de VSCode.
# Usage : bash paper/audit/process/lancer_session_distante.sh <binaire claude> <id de la session a bifurquer>
# Suivre : ssh sur la machine puis `tmux attach -t audit` (detacher : Ctrl-b d), ou Remote Control
# (session "audit-papier-saga-muon" dans claude.ai / l'application mobile).
# Arreter : `tmux kill-session -t audit`.
set -u
BIN="$1"
SID="$2"
PROJ=/users/eleves-b/2023/louis.darrigol/projetSTATS
NAME=audit-papier-saga-muon
PROMPT="La session VSCode te passe le relais : tu es maintenant la session distante de l'audit (tmux 'audit', Remote Control), l'utilisateur est deconnecte. Lis $PROJ/paper/audit/process/consigne_session_distante.md et execute-la en entier, sans poser de question. Commence par lancer le workflow indique."
cd "$PROJ" || exit 1
"$BIN" --remote-control "$NAME" --resume "$SID" --fork-session -n "$NAME" --permission-mode auto "$PROMPT"
echo "[claude termine, code $?] $(date)"
exec bash
