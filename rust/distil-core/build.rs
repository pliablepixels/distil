/// build.rs — link the Python framework when building as rlib/test.
///
/// When `maturin` builds the cdylib it injects the correct Python link flags
/// itself (via pyo3-build-config).  But when `cargo test --lib` builds the
/// rlib for unit-testing, pyo3 needs the Python dylib linked too.
fn main() {
    // Detect if this is a cdylib build driven by maturin.
    let is_maturin = std::env::var("MATURIN_PEP517_ARGS").is_ok();
    if is_maturin {
        return;
    }

    let python = std::env::var("PYO3_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // Query the interpreter for what we need to link.
    // We use a single Python snippet that returns multiple values, one per line.
    let script = r#"
import sysconfig, os, sys

cfg = sysconfig.get_config_vars()
ver = f"{sys.version_info.major}.{sys.version_info.minor}"

# Framework prefix (macOS) — e.g. /opt/homebrew/opt/python@3.14/Frameworks
fw_prefix = cfg.get('PYTHONFRAMEWORKPREFIX', '')

if fw_prefix:
    # Framework build: the actual dylib is Framework/Versions/X.Y/Python
    lib_dir = os.path.join(fw_prefix, 'Python.framework', 'Versions', ver, 'lib')
    lib_name = f"python{ver}"
else:
    lib_dir = cfg.get('LIBDIR', '')
    ldlib = cfg.get('LDLIBRARY', '')
    # strip lib prefix and extension
    lib_name = ldlib
    for prefix in ('lib',):
        if lib_name.startswith(prefix):
            lib_name = lib_name[len(prefix):]
            break
    for suffix in ('.dylib', '.so', '.a'):
        if lib_name.endswith(suffix):
            lib_name = lib_name[:-len(suffix)]
            break

print(lib_dir)
print(lib_name)
"#;

    let out = std::process::Command::new(&python)
        .args(["-c", script])
        .output()
        .expect("failed to run Python to query library info");

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        panic!("Python query failed: {err}");
    }

    let text = String::from_utf8(out.stdout).unwrap();
    let mut lines = text.lines();
    let lib_dir = lines.next().unwrap_or("").trim();
    let lib_name = lines.next().unwrap_or("").trim();

    if !lib_dir.is_empty() && !lib_name.is_empty() {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib=dylib={lib_name}");
    }
}
