use std::env;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::process;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    if args.len() != 2 {
        eprintln!("Usage: cosp <filename>");
        process::exit(1);
    }
    let Ok(mut file) = File::open(&args[1]) else {
        eprintln!("Cannot open file: {}", &args[1]);
        process::exit(1);
    };
    let mut contents = String::new();
    let Ok(_) = file.read_to_string(&mut contents) else {
        eprintln!("Cannot read file: {}", &args[1]);
        process::exit(1);
    };
    let Some(rules) = cosp::parse_rules(&contents) else {
        eprintln!("Cannot parse rules file: {}", &args[1]);
        process::exit(1);
    };
    loop {
        let mut input = String::new();
        print!("?- ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        let Some(query) = cosp::parse_query(input.trim_end()) else {
            eprintln!("Cannot parse input: `{}`", input.trim_end());
            continue;
        };
        let Some((cost, table)) = cosp::infer(&query, &rules) else {
            println!("inf.");
            continue;
        };
        println!("{}{}.", cosp::stringify_table(&table).join(""), cost);
    }
}
