/* Copyright 2025 Northwestern University,
 *                   Carnegie Mellon University University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author(s): David Krasowska <krasow@u.northwestern.edu>
 *            Ethan Meitz <emeitz@andrew.cmu.edu>
 */

#include "legate.h"
#include "types.h"
#include "legion/legion_config.h"


legate::Scalar string_to_scalar(std::string str) {
   return legate::Scalar(str);
}


void wrap_type_enums(jlcxx::Module& mod) {

    auto lt = mod.add_type<legate::Type>("LegateType");
  
    mod.add_bits<legate::Type::Code>("TypeCode", jlcxx::julia_type("CppEnum"));
    mod.set_const("BOOL", legate::Type::Code::BOOL); //legion_type_id_t::LEGION_TYPE_BOOL
    mod.set_const("INT8", legate::Type::Code::INT8); //legion_type_id_t::LEGION_TYPE_INT8
    mod.set_const("INT16", legate::Type::Code::INT16); // legion_type_id_t::LEGION_TYPE_INT16);
    mod.set_const("INT32", legate::Type::Code::INT32); // legion_type_id_t::LEGION_TYPE_INT32);
    mod.set_const("INT64", legate::Type::Code::INT64); // legion_type_id_t::LEGION_TYPE_INT64);
    mod.set_const("UINT8", legate::Type::Code::UINT8); // legion_type_id_t::LEGION_TYPE_UINT8);
    mod.set_const("UINT16", legate::Type::Code::UINT16); // legion_type_id_t::LEGION_TYPE_UINT16);
    mod.set_const("UINT32", legate::Type::Code::UINT32); // legion_type_id_t::LEGION_TYPE_UINT32);
    mod.set_const("UINT64", legate::Type::Code::UINT64); //legion_type_id_t::LEGION_TYPE_UINT64);
    mod.set_const("FLOAT16", legate::Type::Code::FLOAT16); //legion_type_id_t::LEGION_TYPE_FLOAT16);
    mod.set_const("FLOAT32", legate::Type::Code::FLOAT32); // legion_type_id_t::LEGION_TYPE_FLOAT32);
    mod.set_const("FLOAT64", legate::Type::Code::FLOAT64); // legion_type_id_t::LEGION_TYPE_FLOAT64);
    mod.set_const("COMPLEX64", legate::Type::Code::COMPLEX64); // legion_type_id_t::LEGION_TYPE_COMPLEX64);
    mod.set_const("COMPLEX128", legate::Type::Code::COMPLEX128); // legion_type_id_t::LEGION_TYPE_COMPLEX128);
    mod.set_const("NIL", legate::Type::Code::NIL);
    mod.set_const("BINARY", legate::Type::Code::BINARY);
    mod.set_const("FIXED_ARRAY", legate::Type::Code::FIXED_ARRAY);
    mod.set_const("STRUCT", legate::Type::Code::STRUCT);
    mod.set_const("STRING", legate::Type::Code::STRING);
    mod.set_const("LIST", legate::Type::Code::LIST);
  
    lt.method("code", &legate::Type::code);
    lt.method("to_string", &legate::Type::to_string);
    lt.method("string_to_scalar", &string_to_scalar);
  }

  void wrap_type_getters(jlcxx::Module& mod) {
  mod.method("bool_", &legate::bool_);
  mod.method("int8", &legate::int8);
  mod.method("int16", &legate::int16);
  mod.method("int32", &legate::int32);
  mod.method("int64", &legate::int64);
  mod.method("uint8", &legate::uint8);
  mod.method("uint16", &legate::uint16);
  mod.method("uint32", &legate::uint32);
  mod.method("uint64", &legate::uint64);
  mod.method("float16", &legate::float16);
  mod.method("float32", &legate::float32);
  mod.method("float64", &legate::float64);
  // mod.method("complex32", &legate::complex32);
  mod.method("complex64", &legate::complex64);
  mod.method("complex128", &legate::complex128);
}

void wrap_privilege_modes(jlcxx::Module& mod) {
  // from legion_config.h
  mod.add_bits<legion_privilege_mode_t>("LegionPrivilegeMode",
                                        jlcxx::julia_type("CppEnum"));
  mod.set_const("LEGION_READ_ONLY", legion_privilege_mode_t::LEGION_READ_ONLY);
  mod.set_const("LEGION_WRITE_DISCARD",
                legion_privilege_mode_t::LEGION_WRITE_DISCARD);
}
