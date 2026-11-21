use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

fn main() {
    let args = env::args().collect::<Vec<String>>();
    let [_, filename] = args.as_slice() else {
        eprintln!("Usage: cosp <filename>");
        process::exit(1);
    };
    let Ok(contents) = fs::read_to_string(filename) else {
        eprintln!("Cannot read file: {filename}");
        process::exit(1);
    };
    let Ok(rules) = contents.parse() else {
        eprintln!("Cannot parse rules file: {filename}");
        process::exit(1);
    };
    loop {
        let mut input = String::new();
        print!("?- ");
        io::stdout().flush().unwrap();
        if io::stdin().read_line(&mut input).unwrap() == 0 {
            break;
        }
        let input = input.trim_end();
        let Ok(query) = input.parse() else {
            eprintln!("Cannot parse input: `{input}`");
            continue;
        };
        match cosp::infer(&query, &rules) {
            Some((cost, table)) => println!("{table}{cost}."),
            None => println!("inf."),
        }
    }
}
