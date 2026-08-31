def _frame_matches_target(frame, target):
    line_entry = frame.GetLineEntry()
    if not line_entry.IsValid():
        return False

    spec = line_entry.GetFileSpec()
    name = spec.GetFilename() or ''
    dir  = spec.GetDirectory() or ''
    path = f'{dir}/{name}' if dir else name

    return name == target or path.endswith(target)


def print_cadna_backtrace(frame, _bp_loc, dict):
    thread = frame.GetThread()
    target = dict.get('cadna_target_file', '')
    if target and not any(_frame_matches_target(f, target) for f in thread):
        return False

    print('\n================ [CADNA Instability Backtrace] ================', flush=True)
    for i, f in enumerate(thread):
        line_entry = f.GetLineEntry()
        func_name = f.GetFunctionName() or 'unknown'
        filename = line_entry.GetFileSpec().GetFilename() if line_entry.IsValid() else ''
        line_num = line_entry.GetLine() if line_entry.IsValid() else 0

        loc = f'{filename}:{line_num}' if filename else hex(f.GetPC())
        print(f'  #{i:<2} {f.GetModule().GetFileSpec().GetFilename()}`{func_name} at {loc}', flush=True)

    print('===============================================================\n', flush=True)
    return True


def setup_cadna_bp(debugger, command, _result, dict):
    dict['cadna_target_file'] = command.strip()

    debugger.HandleCommand('breakpoint delete cadna_check')
    debugger.HandleCommand('breakpoint set -n instability -N cadna_check')
    debugger.HandleCommand('breakpoint command add -F cadna_lldb.print_cadna_backtrace cadna_check')


def __lldb_init_module(debugger, _dict):
    debugger.HandleCommand('command script add -f cadna_lldb.setup_cadna_bp cadna_setup')
    debugger.HandleCommand('cadna_setup')
