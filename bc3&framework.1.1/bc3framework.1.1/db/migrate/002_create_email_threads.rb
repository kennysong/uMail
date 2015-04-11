class CreateEmailThreads < ActiveRecord::Migration
  def self.up
    create_table :email_threads do |t|
      t.column :id, :primary_key
      t.column :Name, :string
      t.column :Lock, :boolean, :default => false
    end
  end

  def self.down
    drop_table :email_threads
  end
end
