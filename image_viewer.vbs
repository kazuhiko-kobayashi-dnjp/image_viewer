' Image Viewer launcher

Set objShell = CreateObject("WScript.Shell")
Set objFSO   = CreateObject("Scripting.FileSystemObject")

strDir    = objFSO.GetParentFolderName(WScript.ScriptFullName)
strScript = strDir & "\image_viewer.py"

strArgs = ""
If WScript.Arguments.Count > 0 Then
    For Each arg In WScript.Arguments
        strArgs = strArgs & " """ & arg & """"
    Next
End If

Dim pyVersions
pyVersions = Array("3.13", "3.12", "3.11", "3.10")
bLaunched = False

On Error Resume Next
For Each ver In pyVersions
    iRC = objShell.Run("py -" & ver & " -c ""import sys""", 0, True)
    If Err.Number = 0 And iRC = 0 Then
        strCmd = "pyw -" & ver & " """ & strScript & """" & strArgs
        objShell.Run strCmd, 0, False
        bLaunched = True
        Exit For
    End If
    Err.Clear
Next

If Not bLaunched Then
    iRC = objShell.Run("py -3 -c ""import sys""", 0, True)
    If Err.Number = 0 And iRC = 0 Then
        strCmd = "pyw -3 """ & strScript & """" & strArgs
        objShell.Run strCmd, 0, False
        bLaunched = True
    End If
    Err.Clear
End If

If Not bLaunched Then
    MsgBox "Python (py.exe) not found." & vbCrLf & _
           "Please install from https://www.python.org/", _
           vbExclamation, "Image Viewer"
End If
