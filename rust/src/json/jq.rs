use jaq_core::{compile, load, Filter, Native};
use jaq_json::Val;

/// Validates a jq query string by attempting to load and compile it.
/// Returns an error string if the query is invalid.
pub fn build_jq_query(query: &str) -> Result<Filter<Native<Val>>, String> {
    let arena = load::Arena::default();
    let file = load::File {
        path: (),
        code: query,
    };
    let loader = load::Loader::new(jaq_std::defs().chain(jaq_json::defs()));
    let modules = loader
        .load(&arena, file)
        .map_err(|errors| map_load_errors(query, errors))?;

    let compiler = compile::Compiler::default().with_funs(jaq_std::funs().chain(jaq_json::funs()));
    let filter = compiler
        .compile(modules)
        .map_err(|errs| map_compile_errors(query, errs))?;

    Ok(filter)
}

/// Converts jaq load errors into a single error string.
fn map_load_errors(query: &str, errors: load::Errors<&str, ()>) -> String {
    let messages = errors
        .into_iter()
        .flat_map(|(_, err)| match err {
            load::Error::Lex(errs) => errs
                .into_iter()
                .map(|(_, e)| e.to_string())
                .collect::<Vec<_>>(),
            load::Error::Parse(errs) => errs
                .into_iter()
                .map(|(_, e)| e.to_string())
                .collect::<Vec<_>>(),
            load::Error::Io(errs) => errs
                .into_iter()
                .map(|(_, e)| e.to_string())
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();
    format!(
        "Unable to parse jq filter '{query}': {}",
        messages.join(", ")
    )
}

/// Converts jaq compile errors into a single error string.
fn map_compile_errors(query: &str, errs: compile::Errors<&str, ()>) -> String {
    let messages = errs
        .into_iter()
        .flat_map(|(_, errs)| errs.into_iter().map(|(e, _)| e.to_string()))
        .collect::<Vec<_>>();
    format!(
        "Unable to compile jq filter '{query}': {}",
        messages.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_err_contains(query: &str, expected_substring: &str) {
        match build_jq_query(query) {
            Ok(_) => panic!("Expected query to fail, but it succeeded: '{}'", query),
            Err(err) => {
                assert!(
                    err.contains(expected_substring),
                    "Expected error to contain '{}', but got:\n{}",
                    expected_substring,
                    err
                );
            }
        }
    }

    #[test]
    fn validates_identity_query() {
        let query = ".";
        assert!(build_jq_query(query).is_ok());
    }

    #[test]
    fn validates_select_and_map_query() {
        let query = ".[] | select(. > 2) | . * 2";
        assert!(build_jq_query(query).is_ok());
    }

    #[test]
    fn fails_on_parse_error() {
        let query = "?/asdfad"; // clearly invalid syntax
        assert_err_contains(query, "Unable to parse jq filter");
    }

    #[test]
    fn fails_on_compile_error() {
        let query = "foo + 1"; // `foo` is undefined
        assert_err_contains(query, "Unable to compile jq filter");
    }
}
