//! Launch one plugin inside a capability-free Windows AppContainer.
//!
//! Stdio handles are inherited for MCP. No network capability is granted; all
//! network and filesystem access therefore has to return through the host broker.

#[cfg(not(windows))]
fn main() {
    eprintln!("PLUGIN_APPCONTAINER_WINDOWS_REQUIRED");
    std::process::exit(125);
}

#[cfg(windows)]
mod windows_main {
    use std::{
        ffi::c_void,
        mem::size_of,
        os::windows::ffi::OsStrExt,
        path::Path,
        process::{Command, Stdio},
    };

    use windows::{
        core::{PCWSTR, PWSTR},
        Win32::{
            Foundation::{CloseHandle, LocalFree, HLOCAL},
            Security::Isolation::{
                CreateAppContainerProfile, DeriveAppContainerSidFromAppContainerName,
            },
            Security::{Authorization::ConvertSidToStringSidW, FreeSid, SECURITY_CAPABILITIES},
            System::{
                Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE},
                JobObjects::{
                    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
                    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                    JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                    JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_PROCESS_TIME,
                },
                Threading::{
                    CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess,
                    InitializeProcThreadAttributeList, ResumeThread, TerminateProcess,
                    UpdateProcThreadAttribute, WaitForSingleObject, CREATE_NO_WINDOW,
                    CREATE_SUSPENDED, EXTENDED_STARTUPINFO_PRESENT, INFINITE,
                    LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION,
                    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, STARTF_USESTDHANDLES,
                    STARTUPINFOEXW,
                },
            },
        },
    };

    fn wide(value: &str) -> Vec<u16> {
        std::ffi::OsStr::new(value)
            .encode_wide()
            .chain(Some(0))
            .collect()
    }

    fn quote(value: &str) -> String {
        if !value.contains([' ', '\t', '"']) {
            return value.to_string();
        }
        let mut output = String::from("\"");
        let mut slashes = 0;
        for character in value.chars() {
            if character == '\\' {
                slashes += 1;
            } else if character == '"' {
                output.push_str(&"\\".repeat(slashes * 2 + 1));
                output.push('"');
                slashes = 0;
            } else {
                output.push_str(&"\\".repeat(slashes));
                slashes = 0;
                output.push(character);
            }
        }
        output.push_str(&"\\".repeat(slashes * 2));
        output.push('"');
        output
    }

    unsafe fn appcontainer_sid(
        profile: &str,
    ) -> windows::core::Result<windows::Win32::Security::PSID> {
        let profile_wide = wide(profile);
        let display = wide("DroneDream plugin sandbox");
        let description = wide("Capability-free sandbox for a DroneDream plugin");
        match CreateAppContainerProfile(
            PCWSTR(profile_wide.as_ptr()),
            PCWSTR(display.as_ptr()),
            PCWSTR(description.as_ptr()),
            None,
        ) {
            Ok(sid) => Ok(sid),
            Err(_) => DeriveAppContainerSidFromAppContainerName(PCWSTR(profile_wide.as_ptr())),
        }
    }

    unsafe fn sid_string(sid: windows::Win32::Security::PSID) -> Result<String, String> {
        let mut value = PWSTR::null();
        ConvertSidToStringSidW(sid, &mut value)
            .map_err(|_| "PLUGIN_APPCONTAINER_SID_FAILED".to_string())?;
        let result = value
            .to_string()
            .map_err(|_| "PLUGIN_APPCONTAINER_SID_FAILED".to_string());
        LocalFree(Some(HLOCAL(value.0 as *mut c_void)));
        result
    }

    fn grant_read_execute(root: &Path, sid: &str) -> Result<(), String> {
        let status = Command::new("icacls.exe")
            .arg(root)
            .args(["/grant", &format!("*{sid}:(OI)(CI)RX"), "/T", "/C", "/Q"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| "PLUGIN_APPCONTAINER_ACL_FAILED".to_string())?;
        if status.success() {
            Ok(())
        } else {
            Err("PLUGIN_APPCONTAINER_ACL_FAILED".to_string())
        }
    }

    unsafe fn launch(
        profile: &str,
        root: &Path,
        command: &[String],
        memory_mb: usize,
        cpu_seconds: i64,
        process_limit: u32,
    ) -> Result<u32, String> {
        if command.is_empty() {
            return Err("PLUGIN_COMMAND_MISSING".to_string());
        }
        let sid = appcontainer_sid(profile).map_err(|_| "PLUGIN_APPCONTAINER_PROFILE_FAILED")?;
        let sid_text = sid_string(sid)?;
        grant_read_execute(root, &sid_text)?;

        let mut attribute_size = 0usize;
        let _ = InitializeProcThreadAttributeList(None, 1, Some(0), &mut attribute_size);
        let mut attributes = vec![0u8; attribute_size];
        let attribute_list = LPPROC_THREAD_ATTRIBUTE_LIST(attributes.as_mut_ptr().cast());
        InitializeProcThreadAttributeList(Some(attribute_list), 1, Some(0), &mut attribute_size)
            .map_err(|_| "PLUGIN_APPCONTAINER_ATTRIBUTE_FAILED")?;
        let capabilities = SECURITY_CAPABILITIES {
            AppContainerSid: sid,
            Capabilities: std::ptr::null_mut(),
            CapabilityCount: 0,
            Reserved: 0,
        };
        UpdateProcThreadAttribute(
            attribute_list,
            0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES as usize,
            Some(
                (&capabilities as *const SECURITY_CAPABILITIES)
                    .cast_mut()
                    .cast(),
            ),
            size_of::<SECURITY_CAPABILITIES>(),
            None,
            None,
        )
        .map_err(|_| "PLUGIN_APPCONTAINER_ATTRIBUTE_FAILED")?;

        let mut startup = STARTUPINFOEXW::default();
        startup.StartupInfo.cb = size_of::<STARTUPINFOEXW>() as u32;
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput =
            GetStdHandle(STD_INPUT_HANDLE).map_err(|_| "PLUGIN_APPCONTAINER_STDIO_FAILED")?;
        startup.StartupInfo.hStdOutput =
            GetStdHandle(STD_OUTPUT_HANDLE).map_err(|_| "PLUGIN_APPCONTAINER_STDIO_FAILED")?;
        startup.StartupInfo.hStdError =
            GetStdHandle(STD_ERROR_HANDLE).map_err(|_| "PLUGIN_APPCONTAINER_STDIO_FAILED")?;
        startup.lpAttributeList = attribute_list;
        let command_line = command
            .iter()
            .map(|item| quote(item))
            .collect::<Vec<_>>()
            .join(" ");
        let mut command_wide = wide(&command_line);
        let root_wide = wide(&root.to_string_lossy());
        let mut process = PROCESS_INFORMATION::default();
        let created = CreateProcessW(
            PCWSTR::null(),
            Some(PWSTR(command_wide.as_mut_ptr())),
            None,
            None,
            true,
            EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW | CREATE_SUSPENDED,
            None,
            PCWSTR(root_wide.as_ptr()),
            &startup.StartupInfo,
            &mut process,
        );
        DeleteProcThreadAttributeList(attribute_list);
        FreeSid(sid);
        created.map_err(|_| "PLUGIN_APPCONTAINER_PROCESS_FAILED")?;
        let job = match CreateJobObjectW(None, PCWSTR::null()) {
            Ok(value) => value,
            Err(_) => {
                let _ = TerminateProcess(process.hProcess, 125);
                let _ = CloseHandle(process.hThread);
                let _ = CloseHandle(process.hProcess);
                return Err("PLUGIN_RESOURCE_BROKER_START_FAILED".to_string());
            }
        };
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_PROCESS_TIME
            | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        limits.BasicLimitInformation.PerProcessUserTimeLimit = cpu_seconds * 10_000_000;
        limits.BasicLimitInformation.ActiveProcessLimit = process_limit;
        limits.ProcessMemoryLimit = memory_mb * 1024 * 1024;
        let job_ready = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
        .and_then(|_| AssignProcessToJobObject(job, process.hProcess));
        if job_ready.is_err() || ResumeThread(process.hThread) == u32::MAX {
            let _ = TerminateProcess(process.hProcess, 125);
            let _ = CloseHandle(job);
            let _ = CloseHandle(process.hThread);
            let _ = CloseHandle(process.hProcess);
            return Err("PLUGIN_RESOURCE_LIMIT_ASSIGN_FAILED".to_string());
        }
        WaitForSingleObject(process.hProcess, INFINITE);
        let mut exit_code = 125u32;
        GetExitCodeProcess(process.hProcess, &mut exit_code)
            .map_err(|_| "PLUGIN_APPCONTAINER_EXIT_FAILED")?;
        let _ = CloseHandle(process.hThread);
        let _ = CloseHandle(process.hProcess);
        let _ = CloseHandle(job);
        Ok(exit_code)
    }

    pub fn run() -> Result<u32, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        let separator = arguments
            .iter()
            .position(|item| item == "--")
            .ok_or_else(|| "PLUGIN_ISOLATOR_ARGUMENTS_INVALID".to_string())?;
        if separator != 10
            || arguments[0] != "--profile"
            || arguments[2] != "--root"
            || arguments[4] != "--memory-mb"
            || arguments[6] != "--cpu-seconds"
            || arguments[8] != "--process-limit"
        {
            return Err("PLUGIN_ISOLATOR_ARGUMENTS_INVALID".to_string());
        }
        let root = Path::new(&arguments[3]);
        if !root.is_absolute() || !root.is_dir() {
            return Err("PLUGIN_ISOLATOR_ROOT_INVALID".to_string());
        }
        let memory_mb = arguments[5]
            .parse::<usize>()
            .ok()
            .filter(|value| (32..=4096).contains(value))
            .ok_or_else(|| "PLUGIN_RESOURCE_POLICY_INVALID".to_string())?;
        let cpu_seconds = arguments[7]
            .parse::<i64>()
            .ok()
            .filter(|value| (1..=3600).contains(value))
            .ok_or_else(|| "PLUGIN_RESOURCE_POLICY_INVALID".to_string())?;
        let process_limit = arguments[9]
            .parse::<u32>()
            .ok()
            .filter(|value| (1..=64).contains(value))
            .ok_or_else(|| "PLUGIN_RESOURCE_POLICY_INVALID".to_string())?;
        unsafe {
            launch(
                &arguments[1],
                root,
                &arguments[separator + 1..],
                memory_mb,
                cpu_seconds,
                process_limit,
            )
        }
    }
}

#[cfg(windows)]
fn main() {
    match windows_main::run() {
        Ok(code) => std::process::exit(code as i32),
        Err(issue) => {
            eprintln!("{issue}");
            std::process::exit(125);
        }
    }
}
