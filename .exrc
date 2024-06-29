let s:cpo_save=&cpo
set cpo&vim
inoremap <silent> <Plug>(-fzf-complete-finish) l
inoremap <silent> <expr> <Plug>(coc-snippets-expand-jump) coc#_insert_key('request', 'coc-snippets-expand-jump', 1)
inoremap <silent> <expr> <Plug>(coc-snippets-expand) coc#_insert_key('request', 'coc-snippets-expand', 1)
cnoremap <silent> <Plug>(TelescopeFuzzyCommandSearch) e "lua require('telescope.builtin').command_history { default_text = [=[" . escape(getcmdline(), '"') . "]=] }"
inoremap <silent> <F25> :silent doautocmd <nomodeline> FocusGained %
inoremap <silent> <F24> :silent doautocmd <nomodeline> FocusLost %
inoremap <silent> <SNR>69_AutoPairsReturn =AutoPairsReturn()
imap <M-C-Right> <Plug>(copilot-accept-line)
imap <M-Right> <Plug>(copilot-accept-word)
imap <M-Bslash> <Plug>(copilot-suggest)
imap <M-[> <Plug>(copilot-previous)
imap <M-]> <Plug>(copilot-next)
imap <Plug>(copilot-suggest) <Cmd>call copilot#Suggest()
imap <Plug>(copilot-previous) <Cmd>call copilot#Previous()
imap <Plug>(copilot-next) <Cmd>call copilot#Next()
imap <Plug>(copilot-dismiss) <Cmd>call copilot#Dismiss()
inoremap <silent> <expr> <PageUp> coc#pum#visible() ? coc#pum#scroll(0) : "\<PageUp>"
inoremap <silent> <expr> <PageDown> coc#pum#visible() ? coc#pum#scroll(1) : "\<PageDown>"
inoremap <silent> <expr> <C-Y> coc#pum#visible() ? coc#pum#confirm() : "\"
inoremap <silent> <expr> <C-E> coc#pum#visible() ? coc#pum#cancel() : "\"
inoremap <silent> <expr> <Up> coc#pum#visible() ? coc#pum#prev(0) : "\<Up>"
inoremap <silent> <expr> <Down> coc#pum#visible() ? coc#pum#next(0) : "\<Down>"
inoremap <silent> <expr> <C-P> coc#pum#visible() ? coc#pum#prev(1) : "\"
inoremap <silent> <expr> <C-N> coc#pum#visible() ? coc#pum#next(1) : "\"
imap <C-G>S <Plug>ISurround
imap <C-G>s <Plug>Isurround
imap <C-S> <Plug>Isurround
inoremap <silent> <Plug>(fzf-maps-i) :call fzf#vim#maps('i', 0)
inoremap <expr> <Plug>(fzf-complete-buffer-line) fzf#vim#complete#buffer_line()
inoremap <expr> <Plug>(fzf-complete-line) fzf#vim#complete#line()
inoremap <expr> <Plug>(fzf-complete-file-ag) fzf#vim#complete#path('ag -l -g ""')
inoremap <expr> <Plug>(fzf-complete-file) fzf#vim#complete#path("find . -path '*/\.*' -prune -o -type f -print -o -type l -print | sed 's:^..::'")
inoremap <expr> <Plug>(fzf-complete-path) fzf#vim#complete#path("find . -path '*/\.*' -prune -o -print | sed '1d;s:^..::'")
inoremap <expr> <Plug>(fzf-complete-word) fzf#vim#complete#word()
inoremap <nowait> <silent> <expr> <C-B> coc#float#has_scroll() ? "\=coc#float#scroll(0)\" : "\<Left>"
inoremap <nowait> <silent> <expr> <C-F> coc#float#has_scroll() ? "\=coc#float#scroll(1)\" : "\<Right>"
inoremap <silent> <expr> <C-Space> coc#refresh()
inoremap <expr> <S-Tab> coc#pum#visible() ? coc#pum#prev(1) : "\"
inoremap <C-W> u
inoremap <C-U> u
vnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(0) : "\"
nnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(0) : "\"
nnoremap  :WinResizerStartResize
vnoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? coc#float#scroll(1) : "\"
nnoremap  :NERDTreeFind
nnoremap <silent>  :TmuxNavigateLeft
nnoremap <silent> <NL> :TmuxNavigateDown
nnoremap <silent>  :TmuxNavigateUp
nnoremap <silent>  :TmuxNavigateRight
nnoremap  :NERDTree
xmap <silent>  <Plug>(coc-range-select)
nmap <silent>  <Plug>(coc-range-select)
nnoremap  :NERDTreeToggle
nnoremap h s
nnoremap <silent>  :TmuxNavigatePrevious
nnoremap  n :NERDTreeFocus
nnoremap  dh :call vimspector#ToggleBreakpoint()
nnoremap  drc :call vimspector#RunToCursor()
nnoremap  dn :call vimspector#Continue()
nnoremap  d_ :call vimspector#Restart()
nnoremap  dX :call vimspector#ClearBreakpoints()
nnoremap  dx :call vimspector#Reset()
nnoremap  di :call AddToWatch()
nnoremap  do :call GotoWindow(g:vimspector_session_windows.output)
nnoremap  ds :call GotoWindow(g:vimspector_session_windows.stack_trace)
nnoremap  dw :call GotoWindow(g:vimspector_session_windows.watches)
nnoremap  dv :call GotoWindow(g:vimspector_session_windows.variables)
nnoremap  dc :call GotoWindow(g:vimspector_session_windows.code)
nnoremap  db :call vimspector#Launch()
nmap  gt :GTestRunUnderCursor
nmap  cb :CMakeGenerate
nmap  cg :CMakeGenerate
nnoremap <nowait> <silent>  p :CocListResume
nnoremap <nowait> <silent>  k :CocPrev
nnoremap <nowait> <silent>  j :CocNext
nnoremap <nowait> <silent>  s :CocList -I symbols
nnoremap <nowait> <silent>  o :CocList outline
nnoremap <nowait> <silent>  c :CocList commands
nnoremap <nowait> <silent>  e :CocList extensions
nmap  cl <Plug>(coc-codelens-action)
nmap  qf <Plug>(coc-fix-current)
nmap  ac <Plug>(coc-codeaction)
nnoremap <nowait> <silent>  a :CocList diagnostics
xmap  a <Plug>(coc-codeaction-selected)
nmap  F <Plug>(coc-format-selected)
xmap  F <Plug>(coc-format-selected)
nmap  rn <Plug>(coc-rename)
nnoremap  f :Files
nnoremap  tn gt ;new tab
nnoremap  bn :bn ;buffer next
nnoremap   <Nop>
xnoremap # y?\V"
omap <silent> % <Plug>(MatchitOperationForward)
xmap <silent> % <Plug>(MatchitVisualForward)
nmap <silent> % <Plug>(MatchitNormalForward)
xnoremap & : &&
nnoremap & :&&
xnoremap * y/\V"
nnoremap H :call vimspector#StepOut()
nnoremap J :call vimspector#StepOver()
nnoremap <silent> K :call ShowDocumentation()
nnoremap L :call vimspector#StepInto()
xmap S <Plug>VSurround
nnoremap Y y$
omap <silent> [% <Plug>(MatchitOperationMultiBackward)
xmap <silent> [% <Plug>(MatchitVisualMultiBackward)
nmap <silent> [% <Plug>(MatchitNormalMultiBackward)
nmap <silent> [g <Plug>(coc-diagnostic-prev)
omap <silent> ]% <Plug>(MatchitOperationMultiForward)
xmap <silent> ]% <Plug>(MatchitVisualMultiForward)
nmap <silent> ]% <Plug>(MatchitNormalMultiForward)
nmap <silent> ]g <Plug>(coc-diagnostic-next)
xmap a% <Plug>(MatchitVisualTextObject)
omap ac <Plug>(coc-classobj-a)
xmap ac <Plug>(coc-classobj-a)
omap af <Plug>(coc-funcobj-a)
xmap af <Plug>(coc-funcobj-a)
nmap cr <Plug>(abolish-coerce-word)
nmap cS <Plug>CSurround
nmap cs <Plug>Csurround
nmap ds <Plug>Dsurround
xmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
omap <silent> g% <Plug>(MatchitOperationBackward)
xmap <silent> g% <Plug>(MatchitVisualBackward)
nmap <silent> g% <Plug>(MatchitNormalBackward)
nmap gcu <Plug>Commentary<Plug>Commentary
nmap gcc <Plug>CommentaryLine
omap gc <Plug>Commentary
nmap gc <Plug>Commentary
xmap gc <Plug>Commentary
xmap gS <Plug>VgSurround
nmap <silent> gr <Plug>(coc-references)
nmap <silent> gi <Plug>(coc-implementation)
nmap <silent> gy <Plug>(coc-type-definition)
nmap <silent> gd <Plug>(coc-definition)
omap ic <Plug>(coc-classobj-i)
xmap ic <Plug>(coc-classobj-i)
omap if <Plug>(coc-funcobj-i)
xmap if <Plug>(coc-funcobj-i)
nmap ySS <Plug>YSsurround
nmap ySs <Plug>YSsurround
nmap yss <Plug>Yssurround
nmap yS <Plug>YSurround
nmap ys <Plug>Ysurround
nnoremap <silent> <Plug>(-fzf-complete-finish) a
nnoremap <Plug>(-fzf-:) :
nnoremap <Plug>(-fzf-/) /
nnoremap <Plug>(-fzf-vim-do) :execute g:__fzf_command
vnoremap <silent> <Plug>(coc-snippets-select) :call coc#rpc#notify('doKeymap', ['coc-snippets-select'])
xnoremap <silent> <Plug>(coc-convert-snippet) :call coc#rpc#notify('doKeymap', ['coc-convert-snippet'])
tnoremap <silent> <Plug>(fzf-normal) 
tnoremap <silent> <Plug>(fzf-insert) i
nnoremap <silent> <Plug>(fzf-normal) <Nop>
nnoremap <silent> <Plug>(fzf-insert) i
xnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(netrw#GX(),netrw#CheckIfRemote(netrw#GX()))
xmap <silent> <Plug>(MatchitVisualTextObject) <Plug>(MatchitVisualMultiBackward)o<Plug>(MatchitVisualMultiForward)
onoremap <silent> <Plug>(MatchitOperationMultiForward) :call matchit#MultiMatch("W",  "o")
onoremap <silent> <Plug>(MatchitOperationMultiBackward) :call matchit#MultiMatch("bW", "o")
xnoremap <silent> <Plug>(MatchitVisualMultiForward) :call matchit#MultiMatch("W",  "n")m'gv``
xnoremap <silent> <Plug>(MatchitVisualMultiBackward) :call matchit#MultiMatch("bW", "n")m'gv``
nnoremap <silent> <Plug>(MatchitNormalMultiForward) :call matchit#MultiMatch("W",  "n")
nnoremap <silent> <Plug>(MatchitNormalMultiBackward) :call matchit#MultiMatch("bW", "n")
onoremap <silent> <Plug>(MatchitOperationBackward) :call matchit#Match_wrapper('',0,'o')
onoremap <silent> <Plug>(MatchitOperationForward) :call matchit#Match_wrapper('',1,'o')
xnoremap <silent> <Plug>(MatchitVisualBackward) :call matchit#Match_wrapper('',0,'v')m'gv``
xnoremap <silent> <Plug>(MatchitVisualForward) :call matchit#Match_wrapper('',1,'v'):if col("''") != col("$") | exe ":normal! m'" | endifgv``
nnoremap <silent> <Plug>(MatchitNormalBackward) :call matchit#Match_wrapper('',0,'n')
nnoremap <silent> <Plug>(MatchitNormalForward) :call matchit#Match_wrapper('',1,'n')
nnoremap <silent> <C-Bslash> :TmuxNavigatePrevious
nnoremap <silent> <C-K> :TmuxNavigateUp
nnoremap <silent> <C-J> :TmuxNavigateDown
nnoremap <silent> <C-H> :TmuxNavigateLeft
vnoremap <silent> <F25> :silent doautocmd <nomodeline> FocusGained %gv
vnoremap <silent> <F24> :silent doautocmd <nomodeline> FocusLost %gv
onoremap <silent> <F25> :silent doautocmd <nomodeline> FocusGained %
onoremap <silent> <F24> :silent doautocmd <nomodeline> FocusLost %
nnoremap <silent> <F25> :doautocmd <nomodeline> FocusGained %
nnoremap <silent> <F24> :silent doautocmd <nomodeline> FocusLost %
nnoremap <silent> <Plug>(CMakeStop) :call cmake#Stop()
nnoremap <silent> <Plug>(CMakeCloseOverlay) :call cmake#CloseOverlay()
nnoremap <silent> <Plug>(CMakeToggle) :call cmake#Toggle()
nnoremap <silent> <Plug>(CMakeClose) :call cmake#Close(v:false)
nnoremap <silent> <Plug>(CMakeOpen) :call cmake#Open()
nnoremap <silent> <Plug>(CMakeTest) :call cmake#Test()
nnoremap <Plug>(CMakeRun) :CMakeRun 
nnoremap <Plug>(CMakeBuildTarget) :CMakeBuild 
nnoremap <silent> <Plug>(CMakeInstall) :call cmake#Install()
nnoremap <silent> <Plug>(CMakeBuild) :call cmake#Build(v:false)
nnoremap <Plug>(CMakeSwitch) :CMakeSwitch 
nnoremap <silent> <Plug>(CMakeClean) :call cmake#Clean()
nnoremap <silent> <Plug>(CMakeGenerate) :call cmake#Generate(v:false)
nnoremap <Plug>PlenaryTestFile :lua require('plenary.test_harness').test_file(vim.fn.expand("%:p"))
onoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Interface', 'Struct', 'Class'])
vnoremap <silent> <Plug>(coc-classobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Interface', 'Struct', 'Class'])
onoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, '', ['Method', 'Function'])
onoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, '', ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-a) :call CocAction('selectSymbolRange', v:false, visualmode(), ['Method', 'Function'])
vnoremap <silent> <Plug>(coc-funcobj-i) :call CocAction('selectSymbolRange', v:true, visualmode(), ['Method', 'Function'])
nnoremap <silent> <Plug>(coc-cursors-position) :call CocAction('cursorsSelect', bufnr('%'), 'position', 'n')
nnoremap <silent> <Plug>(coc-cursors-word) :call CocAction('cursorsSelect', bufnr('%'), 'word', 'n')
vnoremap <silent> <Plug>(coc-cursors-range) :call CocAction('cursorsSelect', bufnr('%'), 'range', visualmode())
nnoremap <silent> <Plug>(coc-refactor) :call       CocActionAsync('refactor')
nnoremap <silent> <Plug>(coc-command-repeat) :call       CocAction('repeatCommand')
nnoremap <silent> <Plug>(coc-float-jump) :call       coc#float#jump()
nnoremap <silent> <Plug>(coc-float-hide) :call       coc#float#close_all()
nnoremap <silent> <Plug>(coc-fix-current) :call       CocActionAsync('doQuickfix')
nnoremap <silent> <Plug>(coc-openlink) :call       CocActionAsync('openLink')
nnoremap <silent> <Plug>(coc-references-used) :call       CocActionAsync('jumpUsed')
nnoremap <silent> <Plug>(coc-references) :call       CocActionAsync('jumpReferences')
nnoremap <silent> <Plug>(coc-type-definition) :call       CocActionAsync('jumpTypeDefinition')
nnoremap <silent> <Plug>(coc-implementation) :call       CocActionAsync('jumpImplementation')
nnoremap <silent> <Plug>(coc-declaration) :call       CocActionAsync('jumpDeclaration')
nnoremap <silent> <Plug>(coc-definition) :call       CocActionAsync('jumpDefinition')
nnoremap <silent> <Plug>(coc-diagnostic-prev-error) :call       CocActionAsync('diagnosticPrevious', 'error')
nnoremap <silent> <Plug>(coc-diagnostic-next-error) :call       CocActionAsync('diagnosticNext',     'error')
nnoremap <silent> <Plug>(coc-diagnostic-prev) :call       CocActionAsync('diagnosticPrevious')
nnoremap <silent> <Plug>(coc-diagnostic-next) :call       CocActionAsync('diagnosticNext')
nnoremap <silent> <Plug>(coc-diagnostic-info) :call       CocActionAsync('diagnosticInfo')
nnoremap <silent> <Plug>(coc-format) :call       CocActionAsync('format')
nnoremap <silent> <Plug>(coc-rename) :call       CocActionAsync('rename')
nnoremap <Plug>(coc-codeaction-source) :call       CocActionAsync('codeAction', '', ['source'], v:true)
nnoremap <Plug>(coc-codeaction-refactor) :call       CocActionAsync('codeAction', 'cursor', ['refactor'], v:true)
nnoremap <Plug>(coc-codeaction-cursor) :call       CocActionAsync('codeAction', 'cursor')
nnoremap <Plug>(coc-codeaction-line) :call       CocActionAsync('codeAction', 'currline')
nnoremap <Plug>(coc-codeaction) :call       CocActionAsync('codeAction', '')
vnoremap <Plug>(coc-codeaction-refactor-selected) :call       CocActionAsync('codeAction', visualmode(), ['refactor'], v:true)
vnoremap <silent> <Plug>(coc-codeaction-selected) :call       CocActionAsync('codeAction', visualmode())
vnoremap <silent> <Plug>(coc-format-selected) :call       CocActionAsync('formatSelected', visualmode())
nnoremap <Plug>(coc-codelens-action) :call       CocActionAsync('codeLensAction')
nnoremap <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     '', v:true)
vnoremap <silent> <Plug>(coc-range-select-backward) :call       CocActionAsync('rangeSelect',     visualmode(), v:false)
vnoremap <silent> <Plug>(coc-range-select) :call       CocActionAsync('rangeSelect',     visualmode(), v:true)
nmap <silent> <Plug>CommentaryUndo :echoerr "Change your <Plug>CommentaryUndo map to <Plug>Commentary<Plug>Commentary"
nnoremap <silent> <Plug>SurroundRepeat .
onoremap <silent> <Plug>(fzf-maps-o) :call fzf#vim#maps('o', 0)
xnoremap <silent> <Plug>(fzf-maps-x) :call fzf#vim#maps('x', 0)
nnoremap <silent> <Plug>(fzf-maps-n) :call fzf#vim#maps('n', 0)
nnoremap <C-E> :WinResizerStartResize
nnoremap <C-T> :NERDTreeToggle
nnoremap <C-N> :NERDTree
xmap <silent> <C-S> <Plug>(coc-range-select)
nmap <silent> <C-S> <Plug>(coc-range-select)
vnoremap <nowait> <silent> <expr> <C-B> coc#float#has_scroll() ? coc#float#scroll(0) : "\"
vnoremap <nowait> <silent> <expr> <C-F> coc#float#has_scroll() ? coc#float#scroll(1) : "\"
nnoremap <nowait> <silent> <expr> <C-B> coc#float#has_scroll() ? coc#float#scroll(0) : "\"
nnoremap <C-F> :NERDTreeFind
nnoremap <C-W>h s
noremap <Right> <Nop>
noremap <Left> <Nop>
noremap <Down> <Nop>
noremap <Up> <Nop>
nnoremap <silent> <C-L> :TmuxNavigateRight
inoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? "\=coc#float#scroll(0)\" : "\<Left>"
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#cancel() : "\"
inoremap <nowait> <silent> <expr>  coc#float#has_scroll() ? "\=coc#float#scroll(1)\" : "\<Right>"
imap S <Plug>ISurround
imap s <Plug>Isurround
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#confirm(): "\u\\=coc#on_enter()\"
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#next(1) : "\"
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#prev(1) : "\"
imap  <Plug>Isurround
inoremap  u
inoremap  u
inoremap <silent> <expr>  coc#pum#visible() ? coc#pum#confirm() : "\"
cnoremap <expr> %% getcmdtype() == ':' ? expand('%:h').'/' : '%%'
let &cpo=s:cpo_save
unlet s:cpo_save
set clipboard=unnamedplus
set cmdheight=2
set completefunc=tmuxcomplete#complete
set eventignore=CursorHoldI,CursorHold
set expandtab
set grepformat=%f:%l:%c:%m
set grepprg=ack\ --nogroup\ --column\ $*
set helplang=en
set ignorecase
set infercase
set makeprg=make\ PARTY=all\ PROTOCOL=12\ USE_CUDA_GEMM=0\ FUNCTION_IDENTIFIER=23\ NUM_INPUTS=110\ -j
set runtimepath=~/.config/nvim,~/.config/nvim/plugged/csv.vim,~/.config/nvim/plugged/vim-bbye,~/.config/nvim/plugged/winresizer,~/.config/nvim/plugged/fzf.vim,~/.config/nvim/plugged/vim-mundo,~/.config/nvim/plugged/vim-surround,~/.config/nvim/plugged/vim-abolish,~/.config/nvim/plugged/nerdtree,~/.config/nvim/plugged/vim-commentary,~/.config/nvim/plugged/coc.nvim,~/.config/nvim/plugged/nui.nvim,~/.config/nvim/plugged/plenary.nvim,~/.config/nvim/plugged/telescope.nvim,~/.config/nvim/plugged/ChatGPT.nvim,~/.config/nvim/plugged/copilot.vim,~/.config/nvim/plugged/awesome-vim-colorschemes,~/.config/nvim/plugged/vim-devicons,~/.config/nvim/plugged/vim-cpp-enhanced-highlight,~/.config/nvim/plugged/vimspector,~/.config/nvim/plugged/vim-clang-format,~/.config/nvim/plugged/vim-gtest,~/.config/nvim/plugged/vim-cmake,~/.config/nvim/plugged/FixCursorHold.nvim,~/.config/nvim/plugged/synstastic,~/.config/nvim/plugged/auto-pairs,~/.config/nvim/plugged/vim-snippets,~/.config/nvim/plugged/neomake,~/.config/nvim/plugged/nvim-colorizer.lua,~/.config/nvim/plugged/lightline.vim,~/.config/nvim/plugged/tmux-complete.vim,~/.config/nvim/plugged/vim-tmux,~/.config/nvim/plugged/vim-tmux-focus-events,~/.config/nvim/plugged/vim-tmux-navigator,~/.config/nvim/plugged/vim-superman,~/.config/nvim/plugged/which-key.nvim,~/.config/nvim/plugged/ayu-vim,/etc/xdg/nvim,~/.config/local/share/nvim/site,/usr/local/share/nvim/site,/usr/share/nvim/site,/usr/share/nvim/runtime,/usr/share/nvim/runtime/pack/dist/opt/matchit,/usr/lib/nvim,/usr/share/nvim/site/after,/usr/local/share/nvim/site/after,~/.config/local/share/nvim/site/after,/etc/xdg/nvim/after,~/.config/nvim/after,~/.config/nvim/plugged/awesome-vim-colorschemes/after,~/.config/nvim/plugged/vim-cpp-enhanced-highlight/after,~/.config/nvim/plugged/vim-gtest/after,/usr/share/vim/vimfiles,~/.config/coc/extensions/node_modules/coc-snippets
set shiftwidth=4
set smartcase
set softtabstop=4
set statusline=%{coc#status()}%{get(b:,'coc_current_function','')}
set noswapfile
set tabline=%!lightline#tabline()
set tabstop=4
set termguicolors
set undodir=~/.config/nvim/undo
set undofile
set updatetime=300
set window=31
set nowritebackup
" vim: set ft=vim :
