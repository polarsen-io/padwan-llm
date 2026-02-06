# VHS tape file - https://github.com/charmbracelet/vhs
# Run: vhs docs/static/chat-demo.tape

Output docs/static/chat-demo.gif

Set Shell "zsh"
Set FontSize 14
Set Width 800
Set Height 500
Set Theme "Catppuccin Mocha"
Set TypingSpeed 50ms

# Use a clean prompt
Env PROMPT "$ "
Env PS1 "$ "

# Start the TUI
Type "uv run polarsen-llm"
Enter
Sleep 1s

# Type chat:send command
Type "/chat:send "
Sleep 500ms

# Type the message
Type '"What is Python?"'
Sleep 300ms
Enter
Sleep 3s

# Continue the conversation
Type "What are its main features?"
Sleep 300ms
Enter
Sleep 3s

# Exit chat mode
Ctrl+C
Sleep 500ms

# Exit TUI
Ctrl+C
Sleep 300ms
Ctrl+C
Sleep 500ms