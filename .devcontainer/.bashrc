# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
[ -z "$PS1" ] && return

# don't put duplicate lines in the history. See bash(1) for more options
# ... or force ignoredups and ignorespace
HISTCONTROL=ignoredups:ignorespace

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "$debian_chroot" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        # We have color support; assume it's compliant with Ecma-48
        # (ISO/IEC-6429). (Lack of such support is extremely rare, and such
        # a case would tend to support setf rather than setaf.)
        color_prompt=yes
    else
        color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

alias python='python3'
alias pip='pip3'

source /opt/ros/humble/setup.bash 
source /root/ws_moveit/install/setup.bash 

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
#if [ -f /etc/bash_completion ] && ! shopt -oq posix; then
#    . /etc/bash_completion
#fi

# Please echo a message to indicate that the .bashrc has been sourced
echo ""
echo "🤖 ========================================= 🤖"
echo "🚀 Welcome to your ROS 2 Humble + MoveIt Environment!"
echo "🐳 Container: $(hostname)"
echo "📂 Workspace: /ws_lucas"
echo "🤖 ========================================= 🤖"
echo ""

# Display environment status
echo "🔧 Environment Status:"
echo "   🐍 Python: $(python --version 2>/dev/null || echo 'Not found')"
echo "   📦 ROS 2: $(ros2 --version 2>/dev/null | head -1 || echo 'Not configured')"
echo "   🎯 ROS Domain: ${ROS_DOMAIN_ID:-0}"
echo ""

# Quick commands section
echo "⚡ Quick Commands:"
echo "   📋 ros2 topic list           - List all topics"
echo "   🔍 ros2 node list            - List all nodes"
echo "   📊 ros2 pkg list             - List all packages"
echo "   🏥 ros2 doctor                - Check ROS 2 setup"
echo ""

# Demo launches
echo "🎮 Demo Robots & Simulations:"
echo "   🐼 ros2 launch moveit_resources_panda_moveit_config demo.launch.py"
echo "   🦾 ros2 launch moveit2_tutorials demo.launch.py"
echo "   🏭 ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e"
echo ""

# Development commands
echo "🛠️  Development & Testing:"
echo "   🔨 colcon build --packages-select <package_name>"
echo "   🧪 colcon test --packages-select <package_name>"
echo "   🔍 ./test/test_comprehensive.sh"
echo "   📋 ros2 launch --show-args <launch_file>"
echo ""

# Workspace commands
echo "📁 Workspace Management:"
echo "   🔄 source install/setup.bash  - Source workspace"
echo "   🧹 rm -rf build/ install/ log/ - Clean workspace"
echo "   📦 rosdep install --from-paths src --ignore-src -r -y"
echo ""

# Useful aliases reminder
echo "🎯 Custom Aliases Available:"
echo "   📂 ll, la, l                 - Enhanced ls commands"
echo "   🐍 python → python3          - Python 3 default"
echo "   📦 pip → pip3                - Pip 3 default"
echo ""

echo "💡 Tip: Use 'ros2 --help' for more commands!"
echo "🤖 ========================================= 🤖"
echo ""