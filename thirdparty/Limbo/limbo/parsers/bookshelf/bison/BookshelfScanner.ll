/* $Id: scanner.ll 44 2008-10-23 09:03:19Z tb $ -*- mode: c++ -*- */
/** \file scanner.ll Define the example Flex lexical scanner */

%{ /*** C/C++ Declarations ***/

#include <string>

#include "BookshelfScanner.h"

/* import the parser's token type into a local typedef */
typedef BookshelfParser::Parser::token token;
typedef BookshelfParser::Parser::token_type token_type;

/* By default yylex returns int, we use token_type. Unfortunately yyterminate
 * by default returns 0, which is not of token_type. */
#define yyterminate() return token::ENDF

/* This disables inclusion of unistd.h, which is not available under Visual C++
 * on Win32. The C++ scanner uses STL streams instead. */
#define YY_NO_UNISTD_H

%}

/*** Flex Declarations and Options ***/

/* enable c++ scanner class generation */
%option c++

/* change the name of the scanner class. results in "ExampleFlexLexer" */
%option prefix="BookshelfParser"

/* the manual says "somewhat more optimized" */
%option batch

/* enable scanner to generate debug output. disable this for release
 * versions. */
%option debug

/* no support for include files is planned */
%option yywrap nounput 

/* enables the use of start condition stacks */
%option stack

/* The following paragraph suffices to track locations accurately. Each time
 * yylex is invoked, the begin position is moved onto the end position. */
%{
#define YY_USER_ACTION  yylloc->columns(yyleng);
%}

%% /*** Regular Expressions Part ***/

 /* code to place at the beginning of yylex() */
%{
    // reset location
    yylloc->step();
%}

 /*** BEGIN EXAMPLE - Change the example lexer rules below ***/

(?i:X)                  {return token::KWD_X;}
(?i:Y)                  {return token::KWD_Y;}
(?i:H)                  {return token::KWD_H;}
(?i:V)                  {return token::KWD_V;}
(?i:S)                  {return token::KWD_S;}
(?i:Z)                  {return token::KWD_Z;}
(?i:END)                {return token::KWD_END;}
(?i:PIN)                {return token::KWD_PIN;}
(?i:net)                {return token::KWD_NET;}
(?i:CELL)               {return token::KWD_CELL;}
(?i:SITE)               {return token::KWD_SITE;}
(?i:carry)              {return token::KWD_CARRY;}
(?i:MAXSHARED)          {return token::KWD_MAXSHARED;}
(?i:LUTFRACTURE)        {return token::KWD_LUTFRACTURE;}
(?i:SITEDIMENSIONS)     {return token::KWD_SITEDIMENSIONS;}
(?i:SITEPERCOLUMN)      {return token::KWD_SITEPERCOLUMN;}
(?i:CELLDIMENSIONS)     {return token::KWD_CELLDIMENSIONS;}
(?i:SLICEUNIT)          {return token::KWD_SLICEUNIT;}
(?i:FFSLICE)            {return token::KWD_FFSLICE;}
(?i:FFSLICEUNIT)        {return token::KWD_FFSLICEUNIT;}
(?i:FFCTRLS)            {return token::KWD_FFCTRLS;}
(?i:WLWEIGHT)           {return token::KWD_WLWEIGHT;}
(?i:ROUTECAP)           {return token::KWD_ROUTECAP;}
(?i:SITEOUT)            {return token::KWD_SITEOUT;}
(?i:FORMAT)             {return token::KWD_FORMAT;}
(?i:CTRL)               {return token::KWD_CTRL;}
(?i:ADD)                {return token::KWD_ADD;}
(?i:Type)               {return token::KWD_TYPE;}
(?i:INPUT)              {return token::KWD_INPUT;}
(?i:FIXED)              {return token::KWD_FIXED;}
(?i:OUTPUT)             {return token::KWD_OUTPUT;}
(?i:SLICE)              {return token::KWD_SLICE;}
(?i:SLICEL)             {return token::KWD_SLICEL;}
(?i:SLICEM)             {return token::KWD_SLICEM;}
(?i:LAB)                {return token::KWD_LAB;}
(?i:endnet)             {return token::KWD_ENDNET;}
(?i:SITEMAP)            {return token::KWD_SITEMAP;}
(?i:endcarry)           {return token::KWD_ENDCARRY;}
(?i:RESOURCES)          {return token::KWD_RESOURCES;}
(?i:CLOCKREGION)        {return token::KWD_CLOCKREGION;}
(?i:CLOCKREGIONS)       {return token::KWD_CLOCKREGIONS;}
(?i:TYPEINSLICEUNIT)    {return token::KWD_TYPEINSLICEUNIT;}

[A-Za-z0-9_]+\.lib           { yylval->stringVal = new std::string(yytext, yyleng); return token::LIB_FILE; }
[A-Za-z0-9_]+\.scl           { yylval->stringVal = new std::string(yytext, yyleng); return token::SCL_FILE; }
[A-Za-z0-9_]+\.nodes         { yylval->stringVal = new std::string(yytext, yyleng); return token::NODE_FILE; }
[A-Za-z0-9_]+\.nets          { yylval->stringVal = new std::string(yytext, yyleng); return token::NET_FILE; }
[A-Za-z0-9_]+\.cc            { yylval->stringVal = new std::string(yytext, yyleng); return token::CC_FILE; }
[A-Za-z0-9_]+\.pl            { yylval->stringVal = new std::string(yytext, yyleng); return token::PL_FILE; }
[A-Za-z0-9_]+\.wts           { yylval->stringVal = new std::string(yytext, yyleng); return token::WT_FILE; }
[A-Za-z0-9_]+\.lc            { yylval->stringVal = new std::string(yytext, yyleng); return token::LC_FILE; }


[\+\-]?[0-9]+ {
    yylval->integerVal = atol(yytext);
    return token::INTEGER;
}

[\+\-]?[0-9]+\.[0-9]+  {
    yylval->doubleVal = atof(yytext);
    return token::DOUBLE;
}

[\\]?[\_]*[A-Za-z\~][A-Za-z0-9_/\[\]\-\:\|\~\.\;\\]* {
    yylval->stringVal = new std::string(yytext, yyleng);
    return token::STRING;
}

 /* gobble up comments */
"#"[^\n]*                    { yylloc->step(); }

 /* gobble up white-spaces */
[ \t\r]+                     { yylloc->step(); }

 /* gobble up end-of-lines */
\n                           { yylloc->lines(yyleng); yylloc->step(); return token::EOL; }

 /* pass all other characters up to bison */
.                            { return static_cast<token_type>(*yytext); }

 /*** END EXAMPLE - Change the example lexer rules above ***/

%% /*** Additional Code ***/

namespace BookshelfParser {

Scanner::Scanner(std::istream* in, std::ostream* out)
    : BookshelfParserFlexLexer(in, out)
{
}

Scanner::~Scanner()
{
}

void Scanner::set_debug(bool b)
{
    yy_flex_debug = b;
}

}

/* This implementation of ExampleFlexLexer::yylex() is required to fill the
 * vtable of the class ExampleFlexLexer. We define the scanner's main yylex
 * function via YY_DECL to reside in the Scanner class instead. */

#ifdef yylex
#undef yylex
#endif

int BookshelfParserFlexLexer::yylex()
{
    std::cerr << "in BookshelfParserFlexLexer::yylex() !" << std::endl;
    return 0;
}

/* When the scanner receives an end-of-file indication from YY_INPUT, it then
 * checks the yywrap() function. If yywrap() returns false (zero), then it is
 * assumed that the function has gone ahead and set up `yyin' to point to
 * another input file, and scanning continues. If it returns true (non-zero),
 * then the scanner terminates, returning 0 to its caller. */

int BookshelfParserFlexLexer::yywrap()
{
    return 1;
}
