; Welcome-page particle animation for DroneDream AGENT.
; Frames are generated deterministically at build time. The normal sidebar BMP
; remains the fallback when frame loading or animation is unavailable.

!include LogicLib.nsh
!include nsDialogs.nsh

!define AUTONOMY_ANIMATION_FRAME_COUNT 125
!define AUTONOMY_ANIMATION_INTERVAL_MS 40

Var AutonomyAnimationFrame
Var AutonomyAnimationImage
Var AutonomyAnimationBitmap
Var AutonomyWelcomeDialog
Var AutonomyWelcomeTitle

!define MUI_PAGE_CUSTOMFUNCTION_SHOW AutonomyWelcomeAnimationStart
!define MUI_PAGE_CUSTOMFUNCTION_DESTROYED AutonomyWelcomeAnimationStop

Function AutonomyWelcomeAnimationStart
  InitPluginsDir
  SetOutPath "$PLUGINSDIR\autonomy-animation"
  File "${__FILEDIR__}\..\installer\animation\sidebar-frame-*.bmp"
  SetOutPath "$INSTDIR"
  FindWindow $AutonomyWelcomeDialog "#32770" "" $HWNDPARENT
  FindWindow $AutonomyAnimationImage "Static" "" $AutonomyWelcomeDialog
  FindWindow $AutonomyWelcomeTitle "Static" "" $AutonomyWelcomeDialog $AutonomyAnimationImage
  ; 2052 is the NSIS language identifier for Simplified Chinese.
  ${If} $LANGUAGE == 2052
    ${NSD_SetText} $AutonomyWelcomeTitle "安装 DroneDream · AGENT"
  ${Else}
    ${NSD_SetText} $AutonomyWelcomeTitle "Install DroneDream · AGENT"
  ${EndIf}
  StrCpy $AutonomyAnimationFrame 0
  StrCpy $AutonomyAnimationBitmap 0
  ${NSD_CreateTimer} AutonomyWelcomeAnimationTick ${AUTONOMY_ANIMATION_INTERVAL_MS}
FunctionEnd

Function AutonomyWelcomeAnimationTick
  IntFmt $0 "%02d" $AutonomyAnimationFrame
  StrCpy $0 "$PLUGINSDIR\autonomy-animation\sidebar-frame-$0.bmp"

  ${If} $AutonomyAnimationBitmap P<> 0
    ${NSD_FreeImage} $AutonomyAnimationBitmap
  ${EndIf}
  ${NSD_SetStretchedImage} $AutonomyAnimationImage "$0" $AutonomyAnimationBitmap
  IntOp $AutonomyAnimationFrame $AutonomyAnimationFrame + 1
  ${If} $AutonomyAnimationFrame >= ${AUTONOMY_ANIMATION_FRAME_COUNT}
    StrCpy $AutonomyAnimationFrame 0
  ${EndIf}
FunctionEnd

Function AutonomyWelcomeAnimationStop
  ${NSD_KillTimer} AutonomyWelcomeAnimationTick
  ${If} $AutonomyAnimationBitmap P<> 0
    ${NSD_FreeImage} $AutonomyAnimationBitmap
    StrCpy $AutonomyAnimationBitmap 0
  ${EndIf}
FunctionEnd

; Keep the ASCII PRODUCTNAME as the private install/update identity, while the
; user-facing Windows shortcuts use the same canonical name as the title bar
; and approved lockup. Only migrate shortcuts created for this exact install.
!macro AUTONOMY_MIGRATE_DISPLAY_SHORTCUT INTERNAL_PATH DISPLAY_PATH
  ${If} ${FileExists} "${INTERNAL_PATH}"
    ${If} ${FileExists} "${DISPLAY_PATH}"
      !insertmacro IsShortcutTarget "${DISPLAY_PATH}" "$INSTDIR\${MAINBINARYNAME}.exe"
      Pop $0
      ${If} $0 = 1
        Delete "${INTERNAL_PATH}"
        Delete "${DISPLAY_PATH}"
        CreateShortcut "${DISPLAY_PATH}" "$INSTDIR\${MAINBINARYNAME}.exe" "" "$INSTDIR\${MAINBINARYNAME}.exe" 0
      ${EndIf}
    ${Else}
      Delete "${INTERNAL_PATH}"
      CreateShortcut "${DISPLAY_PATH}" "$INSTDIR\${MAINBINARYNAME}.exe" "" "$INSTDIR\${MAINBINARYNAME}.exe" 0
    ${EndIf}
  ${EndIf}
!macroend

!macro AUTONOMY_REMOVE_OWNED_DISPLAY_SHORTCUT DISPLAY_PATH
  ${If} ${FileExists} "${DISPLAY_PATH}"
    !insertmacro IsShortcutTarget "${DISPLAY_PATH}" "$INSTDIR\${MAINBINARYNAME}.exe"
    Pop $0
    ${If} $0 = 1
      Delete "${DISPLAY_PATH}"
    ${EndIf}
  ${EndIf}
!macroend

!macro NSIS_HOOK_POSTINSTALL
  !insertmacro AUTONOMY_MIGRATE_DISPLAY_SHORTCUT "$DESKTOP\${PRODUCTNAME}.lnk" "$DESKTOP\DroneDream · AGENT.lnk"
  !insertmacro AUTONOMY_MIGRATE_DISPLAY_SHORTCUT "$SMPROGRAMS\DroneDream\${PRODUCTNAME}.lnk" "$SMPROGRAMS\DroneDream\DroneDream · AGENT.lnk"
  !insertmacro AUTONOMY_MIGRATE_DISPLAY_SHORTCUT "$SMPROGRAMS\${PRODUCTNAME}.lnk" "$SMPROGRAMS\DroneDream · AGENT.lnk"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  !insertmacro AUTONOMY_REMOVE_OWNED_DISPLAY_SHORTCUT "$DESKTOP\DroneDream · AGENT.lnk"
  !insertmacro AUTONOMY_REMOVE_OWNED_DISPLAY_SHORTCUT "$SMPROGRAMS\DroneDream\DroneDream · AGENT.lnk"
  !insertmacro AUTONOMY_REMOVE_OWNED_DISPLAY_SHORTCUT "$SMPROGRAMS\DroneDream · AGENT.lnk"
!macroend
